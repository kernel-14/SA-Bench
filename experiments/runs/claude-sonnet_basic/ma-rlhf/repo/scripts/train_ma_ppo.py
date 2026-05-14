"""
Main training script for MA-PPO.

Usage:
    python train_ma_ppo.py --config configs/tldr_2b.yaml

This script implements the full MA-RLHF training pipeline:
1. Load SFT model and reward model checkpoints.
2. Initialize policy, reference, critic, and reward models.
3. Run MA-PPO training loop with macro action termination.
"""

import argparse
import json
import logging
import os
import sys
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_utils import (
    TLDRDataset,
    HHRLHFDataset,
    WebGPTDataset,
    APPSDataset,
    split_dataset,
    collate_fn_pad,
)
from ma_ppo_trainer import MAPPOConfig, MAPPOTrainer
from reward_model import RewardModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_dataset(task: str, data: list, tokenizer, mode: str, cfg: dict):
    """Build the appropriate dataset class for the given task."""
    max_prompt = cfg.get("max_prompt_len", 512)
    max_response = cfg.get("max_response_len", 512)

    if task == "tldr":
        return TLDRDataset(data, tokenizer, max_prompt, max_response, mode)
    elif task == "hh_rlhf":
        return HHRLHFDataset(data, tokenizer, max_prompt, max_response, mode)
    elif task == "webgpt":
        return WebGPTDataset(data, tokenizer, max_prompt, max_response, mode)
    elif task == "apps":
        return APPSDataset(data, tokenizer, max_prompt, max_response, mode)
    else:
        raise ValueError(f"Unknown task: {task}")


def load_data(task: str, data_path: str) -> list:
    """Load dataset from a JSONL or JSON file."""
    data = []
    path = Path(data_path)
    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported data format: {path.suffix}")
    return data


def main():
    parser = argparse.ArgumentParser(description="MA-PPO Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info(f"Config: {json.dumps(cfg, indent=2)}")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    task = cfg["task"]
    data = load_data(task, cfg["data_path"])
    logger.info(f"Loaded {len(data)} samples for task '{task}'")

    # Split data: 20% SFT, 40% RM, 40% PPO
    sft_data, rm_data, ppo_data = split_dataset(data)
    logger.info(
        f"Data split: SFT={len(sft_data)}, RM={len(rm_data)}, PPO={len(ppo_data)}"
    )

    # Build PPO dataset
    ppo_dataset = build_dataset(task, ppo_data, tokenizer, "ppo", cfg)
    ppo_loader = DataLoader(
        ppo_dataset,
        batch_size=cfg.get("batch_size", 4),
        shuffle=True,
        collate_fn=partial(collate_fn_pad, pad_token_id=tokenizer.pad_token_id),
    )

    # Load policy model (initialized from SFT checkpoint)
    sft_ckpt = cfg.get("sft_checkpoint", model_name)
    logger.info(f"Loading policy from: {sft_ckpt}")
    policy_model = AutoModelForCausalLM.from_pretrained(
        sft_ckpt, torch_dtype=torch.bfloat16
    ).to(device)

    # Reference model (frozen SFT model)
    ref_model = AutoModelForCausalLM.from_pretrained(
        sft_ckpt, torch_dtype=torch.bfloat16
    ).to(device)
    for param in ref_model.parameters():
        param.requires_grad = False

    # Critic model (initialized from reward model checkpoint)
    rm_ckpt = cfg.get("rm_checkpoint", sft_ckpt)
    logger.info(f"Loading critic/reward model from: {rm_ckpt}")
    critic_base = AutoModelForCausalLM.from_pretrained(
        rm_ckpt, torch_dtype=torch.bfloat16
    ).to(device)
    hidden_size = critic_base.config.hidden_size
    critic_model = RewardModel(critic_base, hidden_size).to(device)

    # Reward model (frozen)
    reward_base = AutoModelForCausalLM.from_pretrained(
        rm_ckpt, torch_dtype=torch.bfloat16
    ).to(device)
    reward_model = RewardModel(reward_base, hidden_size).to(device)
    for param in reward_model.parameters():
        param.requires_grad = False

    # Build MA-PPO config
    ma_config = MAPPOConfig(
        termination=cfg.get("termination", "ngram"),
        n_gram=cfg.get("n_gram", 5),
        parser_cutoff=cfg.get("parser_cutoff", 5),
        value_assignment=cfg.get("value_assignment", "equal"),
        cliprange=cfg.get("clip_ratio", 0.2),
        cliprange_value=cfg.get("clip_ratio", 0.2),
        gamma=cfg.get("gamma", 1.0),
        lam=cfg.get("lam", 0.95),
        kl_coef=cfg.get("kl_coef", 0.05),
        policy_lr=cfg.get("policy_lr", 1.5e-5),
        critic_lr=cfg.get("critic_lr", 1.5e-5),
        batch_size=cfg.get("batch_size", 256),
        ppo_epochs=cfg.get("ppo_epochs", 1),
        max_prompt_len=cfg.get("max_prompt_len", 512),
        max_response_len=cfg.get("max_response_len", 512),
        temperature=cfg.get("temperature", 0.8),
        top_p=cfg.get("top_p", 1.0),
        top_k=cfg.get("top_k", 50),
        warmup_steps=cfg.get("warmup_steps", 200),
        output_dir=cfg.get("output_dir", "./output"),
    )

    # Build trainer
    trainer = MAPPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        critic_model=critic_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        config=ma_config,
    )

    # Training loop
    os.makedirs(ma_config.output_dir, exist_ok=True)
    global_step = 0
    max_steps = cfg.get("max_steps", 5000)

    logger.info(f"Starting MA-PPO training for {max_steps} steps")
    logger.info(f"Termination strategy: {ma_config.termination}, n_gram={ma_config.n_gram}")

    for epoch in range(cfg.get("num_epochs", 100)):
        for batch in ppo_loader:
            if global_step >= max_steps:
                break

            prompts = batch.get("prompt", [])
            if not prompts:
                continue

            # Generate experience
            experience = trainer.generate_experience(prompts)

            # Train step
            metrics = trainer.train_step(experience)
            global_step += 1

            if global_step % ma_config.log_interval == 0:
                logger.info(
                    f"Step {global_step}/{max_steps} | "
                    f"policy_loss={metrics['policy_loss']:.4f} | "
                    f"critic_loss={metrics['critic_loss']:.4f}"
                )

            if global_step % ma_config.save_interval == 0:
                save_path = os.path.join(ma_config.output_dir, f"checkpoint-{global_step}")
                policy_model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                logger.info(f"Saved checkpoint to {save_path}")

        if global_step >= max_steps:
            break

    # Save final model
    final_path = os.path.join(ma_config.output_dir, "final")
    policy_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
