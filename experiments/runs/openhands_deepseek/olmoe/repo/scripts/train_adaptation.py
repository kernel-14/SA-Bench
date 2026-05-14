#!/usr/bin/env python3
"""
Entry point for OLMoE-1B-7B adaptation (SFT + DPO/KTO).

Usage:
    # SFT training
    python scripts/train_adaptation.py --mode sft --config config.yaml \\
        --pretrained_ckpt ./checkpoints/final --sft_data ./data/sft.jsonl

    # DPO training
    python scripts/train_adaptation.py --mode dpo --config config.yaml \\
        --sft_ckpt ./sft_checkpoints/final --dpo_data ./data/dpo.jsonl

    # KTO training
    python scripts/train_adaptation.py --mode kto --config config.yaml \\
        --sft_ckpt ./sft_checkpoints/final --dpo_data ./data/dpo.jsonl

Reproduces adaptation setup from Section 2, 4.3 and Appendix B:
    - SFT: 2 epochs, LR 2e-5, batch 128, no load balancing loss
    - DPO: 3 epochs, LR 5e-7, batch 32, beta 0.1
    - KTO: 1 epoch (5000 steps), RMSProp, same LR/beta
    - Starting from annealed checkpoint
"""
import argparse
import json
import os
import sys
import yaml
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.olmoe_model import OLMoEModel
from training.sft_train import SFTTrainer
from training.dpo_train import DPOTrainer
from data.adaptation_data import create_sft_dataloader, create_dpo_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="OLMoE-1B-7B Adaptation")
    parser.add_argument("--mode", type=str, required=True, choices=["sft", "dpo", "kto"],
                        help="Adaptation mode: sft, dpo, or kto")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--pretrained_ckpt", type=str, default=None,
                        help="Path to pretrained (annealed) checkpoint for SFT")
    parser.add_argument("--sft_ckpt", type=str, default=None,
                        help="Path to SFT checkpoint for DPO/KTO")
    parser.add_argument("--sft_data", type=str, default=None, help="Path to SFT data (JSONL)")
    parser.add_argument("--dpo_data", type=str, default=None, help="Path to DPO/KTO data (JSONL)")
    parser.add_argument("--tokenizer_name", type=str, default="EleutherAI/gpt-neox-20b",
                        help="Tokenizer name or path")
    parser.add_argument("--save_dir", type=str, default="./adaptation_checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Weights & Biases project name")
    return parser.parse_args()


def load_jsonl_data(filepath: str) -> list:
    """Load data from JSONL file."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    model_cfg = config["model"]

    if args.mode == "sft":
        # Load pretrained model (should be annealed checkpoint)
        model = OLMoEModel(
            d_model=model_cfg["d_model"],
            n_layers=model_cfg["n_layers"],
            n_heads=model_cfg["n_heads"],
            vocab_size=model_cfg["vocab_size"],
            max_seq_len=model_cfg["max_seq_len"],
            num_experts=model_cfg["moe"]["num_experts"],
            num_activated_experts=model_cfg["moe"]["num_activated_experts"],
            ffn_dim=model_cfg["moe"]["ffn_dim"],
            qk_norm=model_cfg["qk_norm"],
            layer_norm_eps=model_cfg["layer_norm_eps"],
            rope_theta=model_cfg["rope_theta"],
        )
        if args.pretrained_ckpt:
            checkpoint = torch.load(
                os.path.join(args.pretrained_ckpt, "checkpoint.pt"),
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded pretrained weights from {args.pretrained_ckpt}")
        model.to(device)

        # Load SFT data
        sft_data = load_jsonl_data(args.sft_data)
        print(f"Loaded {len(sft_data)} SFT samples")

        sft_cfg = config["adaptation"]["sft"]
        train_dataloader = create_sft_dataloader(
            data=sft_data,
            tokenizer=tokenizer,
            batch_size=sft_cfg["per_device_batch_size"],
            max_seq_len=sft_cfg["seq_len"],
        )

        trainer = SFTTrainer(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=None,
        )

        trainer.train(
            log_interval=10,
            eval_interval=500,
            save_dir=args.save_dir,
            wandb_project=args.wandb_project,
        )

    elif args.mode in ("dpo", "kto"):
        # Load SFT model as starting point
        model = OLMoEModel(
            d_model=model_cfg["d_model"],
            n_layers=model_cfg["n_layers"],
            n_heads=model_cfg["n_heads"],
            vocab_size=model_cfg["vocab_size"],
            max_seq_len=model_cfg["max_seq_len"],
            num_experts=model_cfg["moe"]["num_experts"],
            num_activated_experts=model_cfg["moe"]["num_activated_experts"],
            ffn_dim=model_cfg["moe"]["ffn_dim"],
            qk_norm=model_cfg["qk_norm"],
            layer_norm_eps=model_cfg["layer_norm_eps"],
            rope_theta=model_cfg["rope_theta"],
        )
        if args.sft_ckpt:
            checkpoint = torch.load(
                os.path.join(args.sft_ckpt, "checkpoint.pt"),
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded SFT weights from {args.sft_ckpt}")
        model.to(device)

        # Reference model for DPO/KTO (frozen copy of SFT model)
        reference_model = OLMoEModel(
            d_model=model_cfg["d_model"],
            n_layers=model_cfg["n_layers"],
            n_heads=model_cfg["n_heads"],
            vocab_size=model_cfg["vocab_size"],
            max_seq_len=model_cfg["max_seq_len"],
            num_experts=model_cfg["moe"]["num_experts"],
            num_activated_experts=model_cfg["moe"]["num_activated_experts"],
            ffn_dim=model_cfg["moe"]["ffn_dim"],
            qk_norm=model_cfg["qk_norm"],
            layer_norm_eps=model_cfg["layer_norm_eps"],
            rope_theta=model_cfg["rope_theta"],
        )
        reference_model.load_state_dict(model.state_dict())
        reference_model.to(device)

        # Load DPO/KTO data
        dpo_data = load_jsonl_data(args.dpo_data)
        print(f"Loaded {len(dpo_data)} preference samples")

        if args.mode == "dpo":
            dpo_cfg = config["adaptation"]["dpo"]
            train_dataloader = create_dpo_dataloader(
                data=dpo_data,
                tokenizer=tokenizer,
                batch_size=dpo_cfg["per_device_batch_size"],
                max_seq_len=4096,
            )

            trainer = DPOTrainer(
                model=model,
                reference_model=reference_model,
                config=config,
                train_dataloader=train_dataloader,
            )

            trainer.train(
                log_interval=10,
                save_dir=args.save_dir,
                wandb_project=args.wandb_project,
            )

        else:  # kto
            # KTO uses same dataloader and trainer as DPO but with different optimizer settings
            kto_cfg = config["adaptation"]["kto"]
            train_dataloader = create_dpo_dataloader(
                data=dpo_data,
                tokenizer=tokenizer,
                batch_size=kto_cfg["per_device_batch_size"],
                max_seq_len=4096,
            )

            trainer = DPOTrainer(
                model=model,
                reference_model=reference_model,
                config=config,
                train_dataloader=train_dataloader,
            )

            trainer.train(
                log_interval=10,
                save_dir=args.save_dir,
                wandb_project=args.wandb_project,
            )

    print(f"Adaptation ({args.mode}) complete!")


if __name__ == "__main__":
    main()
