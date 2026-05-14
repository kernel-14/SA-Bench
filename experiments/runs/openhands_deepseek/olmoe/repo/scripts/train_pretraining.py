#!/usr/bin/env python3
"""
Main entry point for OLMoE-1B-7B pretraining.

Usage:
    python scripts/train_pretraining.py --config config.yaml

Reproduces the pretraining setup from Section 2 and Appendix B:
    - 5.133T tokens total
    - 100B token annealing with linear LR decay to 0
    - Checkpoints saved every 5000 steps
    - All hyperparameters from Table 10
"""
import argparse
import os
import sys
import yaml
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.olmoe_model import OLMoEModel
from training.train import PretrainingTrainer
from data.pretraining_data import create_pretraining_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="OLMoE-1B-7B Pretraining")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--data_paths", type=str, nargs="+", required=True,
                        help="Paths to pretraining data files")
    parser.add_argument("--tokenizer_name", type=str, default="EleutherAI/gpt-neox-20b",
                        help="Tokenizer name or path")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from checkpoint directory")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Weights & Biases project name")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    return parser.parse_args()


def main():
    args = parse_args()

    # Distributed training setup
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{args.local_rank}" if args.local_rank >= 0 else "cuda")

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Create model
    model_cfg = config["model"]
    model = OLMoEModel(
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        vocab_size=model_cfg["vocab_size"],
        max_seq_len=model_cfg["max_seq_len"],
        num_experts=model_cfg["moe"]["num_experts"],
        num_activated_experts=model_cfg["moe"]["num_activated_experts"],
        ffn_dim=model_cfg["moe"]["ffn_dim"],
        dropout=config["pretraining"].get("dropout", 0.0),
        qk_norm=model_cfg["qk_norm"],
        layer_norm_eps=model_cfg["layer_norm_eps"],
        rope_theta=model_cfg["rope_theta"],
    )
    model.to(device)

    # Print model statistics
    active, total = model.get_num_params()
    print(f"Model: {active / 1e9:.2f}B active params, {total / 1e9:.2f}B total params")

    # Distributed model wrapping
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    # Create dataloaders
    train_cfg = config["pretraining"]
    batch_size = train_cfg.get("batch_size_samples", 1024)
    # Adjust batch size for distributed training
    if args.local_rank != -1:
        world_size = dist.get_world_size()
        batch_size = batch_size // world_size

    train_dataloader = create_pretraining_dataloader(
        data_paths=args.data_paths,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_seq_len=train_cfg["max_seq_len"],
        num_workers=4,
        seed=42,
    )

    # Create trainer
    trainer = PretrainingTrainer(
        model=model.module if hasattr(model, "module") else model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=None,
    )

    # Optionally resume from checkpoint
    start_step = 0
    if args.resume_from:
        loaded_config = trainer.load_checkpoint(args.resume_from)
        start_step = int(args.resume_from.split("step_")[-1])

    # Compute total training steps
    total_tokens = train_cfg["total_tokens"]
    tokens_per_step = train_cfg["batch_size_tokens"]
    total_steps = total_tokens // tokens_per_step
    annealing_tokens = train_cfg["annealing_tokens"]
    annealing_steps = annealing_tokens // tokens_per_step

    print(f"Total steps: {total_steps}, Annealing steps: {annealing_steps}")
    print(f"Training for {total_tokens / 1e12:.2f}T tokens")
    print(f"Saving checkpoints every {trainer.save_interval} steps to {args.save_dir}")

    # Train
    trainer.train(
        total_steps=total_steps - start_step,
        annealing_steps=annealing_steps,
        log_interval=100,
        eval_interval=500,
        save_interval=5000,
        save_dir=args.save_dir,
        wandb_project=args.wandb_project,
    )

    print("Pretraining complete!")


if __name__ == "__main__":
    main()
