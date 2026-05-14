"""
Stage 2: Reward Model Training.

Trains a reward model r_φ(x, y) using the Bradley-Terry ranking loss (§2.2):
    L_RM = -log σ(r_φ(x, y+) - r_φ(x, y-))

Initialised from the SFT checkpoint.
Data split: 40% of the full dataset (§B.2).
Not used for APPS (compiler signal replaces RM).

Usage:
    python train_rm.py --task tldr --sft_model_path outputs/sft --output_dir outputs/rm
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from config import TASK_TLDR, TASK_HH_RLHF, TASK_WEBGPT
from data import get_dataset
from model import RewardModel, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reward model training for MA-RLHF")
    parser.add_argument("--task", type=str, default=TASK_TLDR, choices=[TASK_TLDR, TASK_HH_RLHF, TASK_WEBGPT])
    parser.add_argument("--sft_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/rm")
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_response_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    return parser.parse_args()


def get_rm_hyperparams(task: str, model_name: str) -> dict:
    """Return task- and model-specific RM hyperparameters (Table 5)."""
    is_7b = "7b" in model_name.lower()
    is_27b = "27b" in model_name.lower()

    if task == TASK_TLDR:
        return dict(
            batch_size=128 if is_7b else 64,
            epochs=1,
            lr=1e-6 if is_7b else 1e-5,
        )
    elif task == TASK_HH_RLHF:
        return dict(
            batch_size=64,
            epochs=1,
            lr=1e-6 if is_7b else 1e-5,
        )
    elif task == TASK_WEBGPT:
        return dict(
            batch_size=32,
            epochs=32 if is_7b else 1,
            lr=1e-6 if is_7b else 2e-5,
        )
    raise ValueError(f"Unknown task: {task}")


def train_rm(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(args.sft_model_path)
    model = RewardModel(args.sft_model_path)
    model.to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    hp = get_rm_hyperparams(args.task, args.sft_model_path)
    dataset = get_dataset(
        task=args.task,
        stage="rm",
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
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            chosen_reward = model(chosen_ids, chosen_mask)
            rejected_reward = model(rejected_ids, rejected_mask)
            loss = model.ranking_loss(chosen_reward, rejected_reward)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % 100 == 0:
                acc = (chosen_reward > rejected_reward).float().mean().item()
                print(
                    f"Epoch {epoch+1}, Step {global_step}, "
                    f"Loss: {loss.item():.4f}, Acc: {acc:.4f}"
                )

    os.makedirs(args.output_dir, exist_ok=True)
    model.model.save_pretrained(args.output_dir)
    # Save reward head separately
    torch.save(model.reward_head.state_dict(), os.path.join(args.output_dir, "reward_head.pt"))
    tokenizer.save_pretrained(args.output_dir)
    print(f"Reward model saved to {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    train_rm(args)
