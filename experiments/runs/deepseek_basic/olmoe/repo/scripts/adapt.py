"""Adaptation script for OLMoE (SFT + DPO).

Usage:
    # SFT:
    python scripts/adapt.py --mode sft --model_path checkpoint.pt --data_path sft_data.jsonl

    # DPO:
    python scripts/adapt.py --mode dpo --model_path sft_checkpoint.pt --data_path dpo_data.jsonl
"""

import argparse
import os
import sys
from typing import Optional

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from olmoe.models.configuration import OLMoEConfig
from olmoe.models.transformer import OLMoEModel, create_olmoe_model
from olmoe.training.trainer import SFTTrainer, DPOTrainer
from olmoe.data.adaptation import SFTDataset, PreferenceDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Adapt OLMoE via SFT/DPO/KTO")
    parser.add_argument("--mode", type=str, required=True, choices=["sft", "dpo"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./adapted_checkpoints")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--use_lb", action="store_true", help="Use load balancing loss")
    return parser.parse_args()


def run_sft(args):
    """Run Supervised Fine-Tuning."""
    config = OLMoEConfig()

    model = create_olmoe_model()
    if os.path.exists(args.model_path):
        ckpt = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded pretrained checkpoint from {args.model_path}")
    else:
        print(f"Warning: Checkpoint not found at {args.model_path}, using random init")

    # Load data
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    except ImportError:
        class DummyTokenizer:
            pad_token_id = 0
            def encode(self, text):
                return [hash(c) % config.vocab_size for c in text[:1000]]
        tokenizer = DummyTokenizer()

    dataset = SFTDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_seq_len=config.sft_max_seq_len,
    )
    batch_size = args.batch_size or config.sft_batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    trainer = SFTTrainer(model, config, use_load_balancing=args.use_lb)

    epochs = args.epochs or config.sft_epochs
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            metrics = trainer.train_step(
                batch["input_ids"],
                batch["labels"],
            )
            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}: CE Loss = {metrics['ce_loss']:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "sft_checkpoint.pt")
    torch.save({"model_state_dict": model.state_dict()}, out_path)
    print(f"SFT checkpoint saved to {out_path}")


def run_dpo(args):
    """Run Direct Preference Optimization."""
    config = OLMoEConfig()

    model = create_olmoe_model()
    if os.path.exists(args.model_path):
        ckpt = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded SFT checkpoint from {args.model_path}")
    else:
        print(f"Warning: Checkpoint not found at {args.model_path}")

    ref_model = create_olmoe_model()
    ref_model.load_state_dict(model.state_dict())

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    except ImportError:
        class DummyTokenizer:
            pad_token_id = 0
            def encode(self, text):
                return [hash(c) % config.vocab_size for c in text[:1000]]
        tokenizer = DummyTokenizer()

    dataset = PreferenceDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_seq_len=config.sft_max_seq_len,
    )
    batch_size = args.batch_size or config.dpo_batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    trainer = DPOTrainer(model, ref_model, config)

    epochs = args.epochs or config.dpo_epochs
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            metrics = trainer.train_step(
                batch["chosen_input_ids"],
                batch["chosen_labels"],
                batch["rejected_input_ids"],
                batch["rejected_labels"],
            )
            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}: DPO Loss = {metrics['dpo_loss']:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "dpo_checkpoint.pt")
    torch.save({"model_state_dict": model.state_dict()}, out_path)
    print(f"DPO checkpoint saved to {out_path}")


def main():
    args = parse_args()
    if args.mode == "sft":
        run_sft(args)
    elif args.mode == "dpo":
        run_dpo(args)


if __name__ == "__main__":
    main()
