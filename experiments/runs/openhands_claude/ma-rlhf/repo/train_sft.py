"""
Stage 1: Supervised Fine-Tuning (SFT).

Trains the base LM on (prompt, chosen_response) pairs.
Data split: 20% of the full dataset (§B.2).

Usage:
    python train_sft.py --task tldr --model_name google/gemma-2b --output_dir outputs/sft
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from config import SFTConfig, TASK_TLDR, TASK_HH_RLHF, TASK_WEBGPT, TASK_APPS
from data import get_dataset
from model import PolicyModel, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT training for MA-RLHF")
    parser.add_argument("--task", type=str, default=TASK_TLDR, choices=[TASK_TLDR, TASK_HH_RLHF, TASK_WEBGPT, TASK_APPS])
    parser.add_argument("--model_name", type=str, default="google/gemma-2b")
    parser.add_argument("--output_dir", type=str, default="outputs/sft")
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_response_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    return parser.parse_args()


def get_sft_hyperparams(task: str, model_name: str) -> dict:
    """Return task- and model-specific SFT hyperparameters (Table 5)."""
    is_7b = "7b" in model_name.lower()
    is_27b = "27b" in model_name.lower()

    if task == TASK_TLDR:
        return dict(batch_size=128 if is_7b else 512, epochs=3 if not is_7b else 1, lr=2e-5 if is_7b else 5e-5)
    elif task == TASK_HH_RLHF:
        return dict(batch_size=128 if is_7b else 512, epochs=1 if is_7b else 3, lr=2e-5 if is_7b else 5e-5)
    elif task == TASK_WEBGPT:
        return dict(batch_size=64, epochs=5 if is_7b else 3, lr=2e-5 if is_7b else 1e-4)
    elif task == TASK_APPS:
        if is_7b:
            return dict(batch_size=32, epochs=1, lr=2e-6)
        return dict(batch_size=16, epochs=1, lr=5e-6)
    raise ValueError(f"Unknown task: {task}")


def train_sft(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(args.model_name)
    model = PolicyModel(args.model_name)
    model.to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    hp = get_sft_hyperparams(args.task, args.model_name)
    dataset = get_dataset(
        task=args.task,
        stage="sft",
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        seed=args.seed,
    )
    dataloader = DataLoader(dataset, batch_size=hp["batch_size"], shuffle=True, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"])
    total_steps = len(dataloader) * hp["epochs"]
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    global_step = 0
    for epoch in range(hp["epochs"]):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % 100 == 0:
                print(f"Epoch {epoch+1}, Step {global_step}, Loss: {loss.item():.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    model.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"SFT model saved to {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    train_sft(args)
