"""
Reward Model (RM) training script.

Stage 2 of the MA-RLHF pipeline. Trains a reward model on preference pairs
using the Bradley-Terry ranking loss.

Usage:
    python train_rm.py --config configs/tldr_2b.yaml
"""

import argparse
import json
import logging
import os
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_utils import (
    TLDRDataset,
    HHRLHFDataset,
    WebGPTDataset,
    split_dataset,
    collate_fn_pad,
)
from reward_model import RewardModel, reward_model_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(data_path: str) -> list:
    data = []
    path = Path(data_path)
    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        with open(path) as f:
            data = json.load(f)
    return data


def build_rm_dataset(task, data, tokenizer, cfg):
    max_prompt = cfg.get("max_prompt_len", 512)
    max_response = cfg.get("max_response_len", 512)
    if task == "tldr":
        return TLDRDataset(data, tokenizer, max_prompt, max_response, mode="rm")
    elif task == "hh_rlhf":
        return HHRLHFDataset(data, tokenizer, max_prompt, max_response, mode="rm")
    elif task == "webgpt":
        return WebGPTDataset(data, tokenizer, max_prompt, max_response, mode="rm")
    else:
        raise ValueError(f"Unknown task for RM: {task}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = load_data(cfg["data_path"])
    _, rm_data, _ = split_dataset(data)
    logger.info(f"RM data size: {len(rm_data)}")

    task = cfg["task"]
    dataset = build_rm_dataset(task, rm_data, tokenizer, cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("rm_batch_size", 64),
        shuffle=True,
        collate_fn=partial(collate_fn_pad, pad_token_id=tokenizer.pad_token_id),
    )

    # Initialize from SFT checkpoint
    sft_ckpt = cfg.get("sft_checkpoint", model_name)
    logger.info(f"Initializing RM from SFT checkpoint: {sft_ckpt}")
    base_model = AutoModelForCausalLM.from_pretrained(
        sft_ckpt, torch_dtype=torch.bfloat16
    ).to(device)
    hidden_size = base_model.config.hidden_size
    model = RewardModel(base_model, hidden_size).to(device)

    lr = cfg.get("rm_lr", 1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_epochs = cfg.get("rm_epochs", 1)
    total_steps = len(loader) * num_epochs
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    output_dir = cfg.get("rm_output_dir", "./output/rm")
    os.makedirs(output_dir, exist_ok=True)

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        for batch in loader:
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            chosen_rewards = model(chosen_ids, chosen_mask)
            rejected_rewards = model(rejected_ids, rejected_mask)

            loss = reward_model_loss(chosen_rewards, rejected_rewards)
            accuracy = (chosen_rewards > rejected_rewards).float().mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % 100 == 0:
                logger.info(
                    f"Epoch {epoch+1} | Step {global_step} | "
                    f"Loss: {loss.item():.4f} | Acc: {accuracy.item():.4f}"
                )

        logger.info(f"Epoch {epoch+1} complete")

    # Save the base model (without value head) as the RM checkpoint
    model.base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Also save the full reward model state dict
    torch.save(model.state_dict(), os.path.join(output_dir, "reward_model.pt"))
    logger.info(f"Reward model saved to {output_dir}")


if __name__ == "__main__":
    main()
