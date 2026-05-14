"""
Supervised Fine-Tuning (SFT) training script.

Stage 1 of the MA-RLHF pipeline. Fine-tunes a pre-trained LM on human
demonstrations to produce the SFT model used to initialize the policy
and reward model.

Usage:
    python train_sft.py --config configs/tldr_2b.yaml
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
    APPSDataset,
    split_dataset,
    collate_fn_pad,
)

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


def build_dataset(task, data, tokenizer, cfg):
    max_prompt = cfg.get("max_prompt_len", 512)
    max_response = cfg.get("max_response_len", 512)
    if task == "tldr":
        return TLDRDataset(data, tokenizer, max_prompt, max_response, mode="sft")
    elif task == "hh_rlhf":
        return HHRLHFDataset(data, tokenizer, max_prompt, max_response, mode="sft")
    elif task == "webgpt":
        return WebGPTDataset(data, tokenizer, max_prompt, max_response, mode="sft")
    elif task == "apps":
        return APPSDataset(data, tokenizer, max_prompt, max_response, mode="sft")
    else:
        raise ValueError(f"Unknown task: {task}")


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
    sft_data, _, _ = split_dataset(data)
    logger.info(f"SFT data size: {len(sft_data)}")

    task = cfg["task"]
    dataset = build_dataset(task, sft_data, tokenizer, cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("sft_batch_size", 64),
        shuffle=True,
        collate_fn=partial(collate_fn_pad, pad_token_id=tokenizer.pad_token_id),
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)

    lr = cfg.get("sft_lr", 5e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_epochs = cfg.get("sft_epochs", 3)
    total_steps = len(loader) * num_epochs
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    output_dir = cfg.get("sft_output_dir", "./output/sft")
    os.makedirs(output_dir, exist_ok=True)

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % 100 == 0:
                logger.info(f"Epoch {epoch+1} | Step {global_step} | Loss: {loss.item():.4f}")

        logger.info(f"Epoch {epoch+1} complete")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"SFT model saved to {output_dir}")


if __name__ == "__main__":
    main()
