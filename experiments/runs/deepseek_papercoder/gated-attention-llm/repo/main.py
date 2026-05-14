## main.py
"""
Entry point for reproducing Gated Attention LLM experiments.

Supports four subcommands:
    train   – train a model from scratch.
    eval    – evaluate a trained checkpoint (PPL + downstream benchmarks).
    analyze – run gate sparsity, attention sink, and massive activation analysis.
    all     – sequentially train (unless --skip-train), evaluate, and analyse.

Usage examples:
    python main.py train --config config.yaml
    python main.py eval  --config config.yaml --resume-checkpoint checkpoints/final/model_state_dict.pt
    python main.py all   --config config.yaml [--skip-train]
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import transformers
from accelerate import Accelerator
from transformers import AutoTokenizer

import utils
from data import DataModule
from model import GPTModel
from trainer import Trainer, ConsoleLogger, Logger
from eval import Evaluator
from analysis import Analyzer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Gated Attention LLM Reproduction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---------- train ----------
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", type=str, required=True, help="Path to YAML config")

    # ---------- eval ----------
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    eval_parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    eval_parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to model state dict (.pt). If omitted, attempts to use checkpoint/final/model_state_dict.pt",
    )

    # ---------- analyse ----------
    analyse_parser = subparsers.add_parser("analyze", help="Analyse internal statistics")
    analyse_parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    analyse_parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to model state dict (.pt). If omitted, attempts to use checkpoint/final/model_state_dict.pt",
    )
    analyse_parser.add_argument(
        "--output-dir",
        type=str,
        default="./analysis_output",
        help="Directory to save analysis results (CSV/plots).",
    )

    # ---------- all ----------
    all_parser = subparsers.add_parser("all", help="Train + eval + analyse")
    all_parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    all_parser.add_argument("--skip-train", action="store_true", help="Skip training (use existing checkpoint)")
    all_parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to model state dict (.pt) if skipping training. Otherwise uses default after training.",
    )
    all_parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiment_output",
        help="Root directory for all outputs (training, evaluation, analysis).",
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# Helper: instantiate model
# ----------------------------------------------------------------------
def build_model(config: Dict, device: torch.device) -> GPTModel:
    """
    Create a GPTModel instance, applying FFN width adjustment if gating is enabled.

    Args:
        config: Full configuration dictionary.
        device: Device to place the model on.

    Returns:
        Instantiated GPTModel on the target device.
    """
    model_cfg = config["model"]
    # Validate GQA compatibility
    if model_cfg["num_attention_heads"] % model_cfg["num_key_value_heads"] != 0:
        raise ValueError(
            f"num_attention_heads ({model_cfg['num_attention_heads']}) must be divisible by "
            f"num_key_value_heads ({model_cfg['num_key_value_heads']}) for Grouped‑Query Attention."
        )

    # Adjust FFN intermediate size if gating is used (parameter matching)
    if model_cfg.get("use_gated_attention", False):
        original_intermediate = model_cfg["intermediate_size"]
        new_intermediate = utils.adjust_intermediate_size(config)
        model_cfg["intermediate_size"] = new_intermediate
        if new_intermediate == original_intermediate:
            print(
                "Note: Gate parameters are zero or FFN width unchanged – parameter count may not be matched."
            )
        else:
            print(
                f"Adjusted intermediate_size from {original_intermediate} to {new_intermediate} "
                f"to compensate for gating parameters."
            )

    model = GPTModel(model_cfg)
    model.to(device)
    return model


# ----------------------------------------------------------------------
# Helper: load checkpoint
# ----------------------------------------------------------------------
def load_checkpoint(model: GPTModel, checkpoint_path: str, device: torch.device) -> None:
    """
    Load a saved state_dict into the model.

    Args:
        model: GPTModel instance (architecture must match the checkpoint).
        checkpoint_path: Path to a .pt file containing a state_dict.
        device: Target device.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    # Some checkpoints may have been saved with accelerator wrap; use the state directly.
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: unexpected keys in checkpoint: {unexpected_keys}")
    print(f"Successfully loaded checkpoint from {checkpoint_path}")


# ----------------------------------------------------------------------
# Main dispatch
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    command = args.command

    # 1. Load configuration
    config_path = args.config
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config = utils.load_config(config_path)

    # 2. Set random seed
    training_cfg = config.get("training", {})
    seed = training_cfg.get("seed", 42)
    utils.set_seed(seed)

    # 3. Determine device (we use CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4. Load tokenizer
    data_cfg = config.get("data", {})
    tokenizer_path = data_cfg.get("tokenizer_name_or_path", "")
    if not tokenizer_path:
        raise ValueError("Tokenizer path not specified in config.data.tokenizer_name_or_path")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    # Ensure the tokenizer has a pad token; set to eos if missing (common for causal LLM)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    vocab_size = len(tokenizer)
    model_cfg = config["model"]
    if model_cfg["vocab_size"] != vocab_size:
        print(
            f"Warning: config.model.vocab_size ({model_cfg['vocab_size']}) differs from "
            f"tokenizer vocab size ({vocab_size}). Overriding config."
        )
        model_cfg["vocab_size"] = vocab_size

    # 5. Paths for checkpoint and output
    root_output = getattr(args, "output_dir", "./experiment_output")
    checkpoint_dir = os.path.join(root_output, "checkpoints")
    checkpoint_path = getattr(args, "resume_checkpoint", None)  # may be None

    # ------------------------------------------------------------------
    # Branch: train
    # ------------------------------------------------------------------
    if command == "train":
        # Build model and training infrastructure
        model = build_model(config, device)

        datamodule = DataModule(
            config=config,
            tokenizer=tokenizer,
            seq_length=training_cfg.get("seq_length", model_cfg["max_position_embeddings"]),
            batch_size=training_cfg.get("global_batch_size", 1024) // torch.cuda.device_count() if torch.cuda.is_available() else training_cfg.get("global_batch_size", 1024),  # rough per-device batch
        )

        # Logger: if wandb is available, use it; otherwise print to console.
        use_wandb = config.get("logging", {}).get("use_wandb", False)
        wandb_project = config.get("logging", {}).get("wandb_project", "gated-attention-llm")
        logger = ConsoleLogger(use_wandb=use_wandb, wandb_project=wandb_project)

        training_config = config.get("training", {})
        # Add output directory to training config
        training_config["output_dir"] = checkpoint_dir

        trainer = Trainer(
            model=model,
            datamodule=datamodule,
            config=training_config,
            logger=logger,
        )
        trainer.train()

    # ------------------------------------------------------------------
    # Branch: eval
    # ------------------------------------------------------------------
    elif command == "eval":
        model = build_model(config, device)

        # Determine checkpoint to load
        if checkpoint_path is None:
            # default location: checkpoint/final/model_state_dict.pt
            default_ckpt = os.path.join(checkpoint_dir, "final", "model_state_dict.pt")
            if not os.path.isfile(default_ckpt):
                raise FileNotFoundError(
                    f"No checkpoint provided and default {default_ckpt} does not exist. "
                    "Use --resume-checkpoint."
                )
            checkpoint_path = default_ckpt
        load_checkpoint(model, checkpoint_path, device)

        eval_cfg = config.get("evaluation", {})
        evaluator = Evaluator(model, eval_cfg, tokenizer)

        # Run perplexity on eval dataloaders
        per_domain_ppl = {}
        datamodule = DataModule(config, tokenizer, seq_length=model_cfg["max_position_embeddings"], batch_size=1)
        for domain in data_cfg.get("eval_datasets", []):
            print(f"Computing PPL for domain: {domain}")
            dl = datamodule.get_eval_dataloader(domain)
            ppl = evaluator.compute_perplexity(dl)
            per_domain_ppl[domain] = ppl
            print(f"  {domain}: PPL = {ppl:.4f}")

        avg_ppl = sum(per_domain_ppl.values()) / len(per_domain_ppl) if per_domain_ppl else float("nan")
        print(f"Average PPL across domains: {avg_ppl:.4f}")

        # Downstream benchmarks
        downstream_scores = evaluator._run_lm_eval_tasks()
        print("Downstream task scores:")
        for task, score in downstream_scores.items():
            print(f"  {task}: {score:.4f}")

        # Save results
        results = {
            "avg_ppl": avg_ppl,
            "per_domain_ppl": per_domain_ppl,
            "downstream": downstream_scores,
        }
        os.makedirs(root_output, exist_ok=True)
        with open(os.path.join(root_output, "eval_results.json"), "w") as f:
            import json
            json.dump(results, f, indent=2)
        print(f"Evaluation results saved to {root_output}/eval_results.json")

    # ------------------------------------------------------------------
    # Branch: analyse
    # ------------------------------------------------------------------
    elif command == "analyze":
        model = build_model(config, device)
        if checkpoint_path is None:
            default_ckpt = os.path.join(checkpoint_dir, "final", "model_state_dict.pt")
            if not os.path.isfile(default_ckpt):
                raise FileNotFoundError(
                    f"No checkpoint provided and default {default_ckpt} does not exist. "
                    "Use --resume-checkpoint."
                )
            checkpoint_path = default_ckpt
        load_checkpoint(model, checkpoint_path, device)

        output_dir = args.output_dir if hasattr(args, "output_dir") else "./analysis_output"
        os.makedirs(output_dir, exist_ok=True)

        datamodule = DataModule(config, tokenizer, seq_length=model_cfg["max_position_embeddings"], batch_size=1)
        # Use a small validation dataset for analysis (e.g., the first eval domain)
        if not data_cfg.get("eval_datasets"):
            raise ValueError("At least one eval dataset must be specified for analysis.")
        analysis_domain = data_cfg["eval_datasets"][0]
        analysis_dl = datamodule.get_eval_dataloader(analysis_domain)
        # For memory, limit to the first 10 batches
        limited_dl = []
        for i, batch in enumerate(analysis_dl):
            if i >= 10:
                break
            limited_dl.append(batch)

        analyzer = Analyzer(model, config)

        # 1. Gate sparsity
        gate_stats = analyzer.analyze_gate_sparsity(limited_dl)
        # 2. Attention sink
        attn_sink = analyzer.analyze_attention_sink(limited_dl)
        # 3. Massive activations
        massive_act = analyzer.analyze_massive_activations(limited_dl)

        # Save and print
        import numpy as np
        import json

        results = {
            "mean_gate_scores": gate_stats["mean_gate_scores"],
            "sparsity": gate_stats["sparsity"],
            "f_attn": attn_sink["f_attn"],
            "m_act": massive_act,
        }
        with open(os.path.join(output_dir, "analysis.json"), "w") as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        print("Analysis results saved to", os.path.join(output_dir, "analysis.json"))
        # Print a quick summary
        print("\n--- Gate Sparsity (mean per layer) ---")
        for i, m in enumerate(gate_stats["mean_gate_scores"]):
            print(f"Layer {i:2d}: mean={m:.4f}")
        print("\n--- Attention to first token (F‑Attn) ---")
        for i, f in enumerate(attn_sink["f_attn"]):
            print(f"Layer {i:2d}: first_token_attn={f:.4f}")
        print("\n--- Massive Activations (M‑Act) ---")
        for i, m in enumerate(massive_act):
            print(f"Layer {i:2d}: max_act_mean={m:.4f}")

    # ------------------------------------------------------------------
    # Branch: all (train + eval + analyse)
    # ------------------------------------------------------------------
    elif command == "all":
        skip_train = args.skip_train
        resume_checkpoint = args.resume_checkpoint
        output_dir = args.output_dir
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(output_dir, exist_ok=True)

        # ---------- 1. Training ----------
        if not skip_train:
            print("Starting training...")
            model = build_model(config, device)
            datamodule = DataModule(
                config=config,
                tokenizer=tokenizer,
                seq_length=training_cfg.get("seq_length", model_cfg["max_position_embeddings"]),
                batch_size=training_cfg.get("global_batch_size", 1024) // torch.cuda.device_count() if torch.cuda.is_available() else training_cfg.get("global_batch_size", 1024),
            )
            use_wandb = config.get("logging", {}).get("use_wandb", False)
            wandb_project = config.get("logging", {}).get("wandb_project", "gated-attention-llm")
            logger = ConsoleLogger(use_wandb=use_wandb, wandb_project=wandb_project)
            training_config = config.get("training", {})
            training_config["output_dir"] = checkpoint_dir
            trainer = Trainer(
                model=model,
                datamodule=datamodule,
                config=training_config,
                logger=logger,
            )
            trainer.train()
            # After training, the final checkpoint is at checkpoint_dir/final/model_state_dict.pt
            resume_checkpoint = os.path.join(checkpoint_dir, "final", "model_state_dict.pt")
        else:
            if resume_checkpoint is None:
                # Try default location under output_dir
                default_ckpt = os.path.join(checkpoint_dir, "final", "model_state_dict.pt")
                if not os.path.isfile(default_ckpt):
                    raise FileNotFoundError(
                        "Skip train specified but no checkpoint provided and default not found. "
                        "Use --resume-checkpoint."
                    )
                resume_checkpoint = default_ckpt

        # ---------- 2. Evaluation ----------
        print("Starting evaluation...")
        eval_model = build_model(config, device)
        load_checkpoint(eval_model, resume_checkpoint, device)

        eval_cfg = config.get("evaluation", {})
        evaluator = Evaluator(eval_model, eval_cfg, tokenizer)

        per_domain_ppl = {}
        datamodule_eval = DataModule(config, tokenizer, seq_length=model_cfg["max_position_embeddings"], batch_size=1)
        for domain in data_cfg.get("eval_datasets", []):
            print(f"PPL for domain: {domain}")
            dl = datamodule_eval.get_eval_dataloader(domain)
            ppl = evaluator.compute_perplexity(dl)
            per_domain_ppl[domain] = ppl
            print(f"  {domain}: {ppl:.4f}")

        avg_ppl = sum(per_domain_ppl.values()) / len(per_domain_ppl) if per_domain_ppl else float("nan")
        print(f"Average PPL: {avg_ppl:.4f}")

        downstream_scores = evaluator._run_lm_eval_tasks()
        eval_results = {
            "avg_ppl": avg_ppl,
            "per_domain_ppl": per_domain_ppl,
            "downstream": downstream_scores,
        }
        eval_out_path = os.path.join(output_dir, "eval_results.json")
        with open(eval_out_path, "w") as f:
            import json
            json.dump(eval_results, f, indent=2)
        print("Evaluation results saved.")

        # ---------- 3. Analysis ----------
        print("Starting analysis...")
        analysis_model = build_model(config, device)
        load_checkpoint(analysis_model, resume_checkpoint, device)

        analysis_output_dir = os.path.join(output_dir, "analysis")
        os.makedirs(analysis_output_dir, exist_ok=True)

        datamodule_analysis = DataModule(config, tokenizer, seq_length=model_cfg["max_position_embeddings"], batch_size=1)
        analysis_domain = data_cfg["eval_datasets"][0]
        analysis_dl = datamodule_analysis.get_eval_dataloader(analysis_domain)
        # limit to first 10 batches
        limited_dl = []
        for i, batch in enumerate(analysis_dl):
            if i >= 10:
                break
            limited_dl.append(batch)

        analyzer = Analyzer(analysis_model, config)
        gate_stats = analyzer.analyze_gate_sparsity(limited_dl)
        attn_sink = analyzer.analyze_attention_sink(limited_dl)
        massive_act = analyzer.analyze_massive_activations(limited_dl)

        analysis_results = {
            "mean_gate_scores": gate_stats["mean_gate_scores"],
            "sparsity": gate_stats["sparsity"],
            "f_attn": attn_sink["f_attn"],
            "m_act": massive_act,
        }
        import numpy as np
        import json
        with open(os.path.join(analysis_output_dir, "analysis.json"), "w") as f:
            json.dump(analysis_results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        print("Analysis results saved.")

    else:
        raise RuntimeError(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
