"""Full training pipeline for MA-RLHF.

Implements the three-stage post-training paradigm (§2.2):
1. Supervised Fine-Tuning (SFT)
2. Reward Modeling (RM)
3. RLHF with PPO / MA-PPO

Supports datasets: TL;DR, HH-RLHF, WebGPT, APPS
Supports models: Gemma-2B, 7B, 27B; CodeGemma-2B, 7B
"""
import os
import sys
import json
import yaml
import argparse
import logging
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import wandb

from config import (
    ExperimentConfig, SFTConfig, RMConfig, PPOConfig, MAPPOConfig,
    get_gemma_2b_config, get_gemma_7b_config, get_gemma_27b_config,
    get_codegemma_2b_config, get_codegemma_7b_config, CONFIG_MAP,
)
from data import (
    get_dataset, collate_for_rlhf,
    SFTDataset, PreferenceDataset, RLHFDataset, APPSDataset,
)
from model import SFTModel, RewardModel, PolicyModel, CriticModel, ReferenceModel
from ppo import VanillaPPO, MAPPO
from macro_actions import compute_perplexity_values

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def train_sft(
    model: SFTModel,
    dataset: SFTDataset,
    config: SFTConfig,
    tokenizer,
    output_dir: str,
    device: torch.device,
):
    """Stage 1: Supervised Fine-Tuning."""
    logger.info("Starting SFT training...")
    model.train()
    model.base_model.to(device)

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, drop_last=True,
    )

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(dataloader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    global_step = 0
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"SFT Epoch {epoch + 1}/{config.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, _ = model(input_ids, attention_mask, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"SFT Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

    checkpoint_dir = os.path.join(output_dir, "sft_model")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    logger.info(f"SFT model saved to {checkpoint_dir}")
    return checkpoint_dir


def train_rm(
    model: RewardModel,
    dataset: PreferenceDataset,
    config: RMConfig,
    output_dir: str,
    device: torch.device,
):
    """Stage 2: Reward Modeling.

    Trains reward model using ranking loss:
    L_RM = -log σ(r(x, y+) - r(x, y-))
    """
    logger.info("Starting Reward Model training...")
    model.train()
    model.base_model.to(device)
    model.value_head.to(device)

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, drop_last=True,
    )

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(dataloader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        pbar = tqdm(dataloader, desc=f"RM Epoch {epoch + 1}/{config.epochs}")
        for batch in pbar:
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            r_chosen = model(chosen_ids, chosen_mask)
            r_rejected = model(rejected_ids, rejected_mask)

            # Ranking loss
            loss = -torch.nn.functional.logsigmoid(r_chosen - r_rejected).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            acc = (r_chosen > r_rejected).float().mean().item()
            epoch_acc += acc
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.3f}"})

        avg_loss = epoch_loss / len(dataloader)
        avg_acc = epoch_acc / len(dataloader)
        logger.info(f"RM Epoch {epoch + 1} avg loss: {avg_loss:.4f}, acc: {avg_acc:.4f}")

    checkpoint_dir = os.path.join(output_dir, "rm_model")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    logger.info(f"RM model saved to {checkpoint_dir}")
    return checkpoint_dir


def train_rlhf(
    ppo_trainer,
    dataset: RLHFDataset,
    ppo_config: PPOConfig,
    output_dir: str,
    device: torch.device,
    max_steps: int = 5000,
    save_every: int = 500,
    eval_dataset: Optional[RLHFDataset] = None,
    eval_every: int = 500,
    use_wandb: bool = False,
    method_name: str = "PPO",
):
    """Stage 3: RLHF with PPO or MA-PPO."""
    logger.info(f"Starting {method_name} training...")

    ppo_trainer.policy_model.base_model.to(device)
    ppo_trainer.critic_model.base_model.to(device)
    ppo_trainer.critic_model.value_head.to(device)
    ppo_trainer.reference_model.base_model.to(device)
    ppo_trainer.reward_model.base_model.to(device)
    ppo_trainer.reward_model.value_head.to(device)

    dataloader = DataLoader(
        dataset, batch_size=ppo_config.batch_size, shuffle=True, drop_last=True,
        collate_fn=collate_for_rlhf,
    )

    policy_optimizer = AdamW(
        ppo_trainer.policy_model.parameters(), lr=ppo_config.policy_learning_rate,
    )
    critic_optimizer = AdamW(
        ppo_trainer.critic_model.parameters(), lr=ppo_config.critic_learning_rate,
    )

    global_step = 0
    metrics_history = []

    # Create infinite dataloader for RL
    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            # Single PPO step
            metrics = ppo_trainer.step(batch)
            policy_loss = metrics["policy_loss"]
            value_loss = metrics["value_loss"]
            total_loss = policy_loss + value_loss

            # Update policy
            policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(ppo_trainer.policy_model.parameters(), 0.5)
            policy_optimizer.step()

            # Update critic (if separate optimizer)
            critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(ppo_trainer.critic_model.parameters(), 0.5)
            critic_optimizer.step()

            global_step += 1
            metrics["step"] = global_step
            metrics_history.append(metrics)

            if use_wandb:
                wandb.log(metrics, step=global_step)

            if global_step % 10 == 0:
                log_str = (
                    f"{method_name} Step {global_step} | "
                    f"policy_loss: {policy_loss.item():.4f} | "
                    f"value_loss: {value_loss.item():.4f} | "
                    f"rm_reward: {metrics['rm_reward_mean']:.4f}"
                )
                if "num_macro_actions" in metrics:
                    log_str += f" | ma: {metrics['num_macro_actions']}"
                logger.info(log_str)

            if global_step % save_every == 0:
                ckpt_dir = os.path.join(output_dir, f"{method_name}_step{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                ppo_trainer.policy_model.save_pretrained(os.path.join(ckpt_dir, "policy"))
                ppo_trainer.critic_model.save_pretrained(os.path.join(ckpt_dir, "critic"))
                logger.info(f"Checkpoint saved at step {global_step}")

    # Save final model
    final_dir = os.path.join(output_dir, f"{method_name}_final")
    os.makedirs(final_dir, exist_ok=True)
    ppo_trainer.policy_model.save_pretrained(os.path.join(final_dir, "policy"))
    ppo_trainer.critic_model.save_pretrained(os.path.join(final_dir, "critic"))

    # Save metrics
    with open(os.path.join(output_dir, f"{method_name}_metrics.json"), "w") as f:
        json.dump(metrics_history, f, indent=2)

    logger.info(f"{method_name} training complete. Final model at {final_dir}")
    return final_dir


def train_sft_only(config: ExperimentConfig, device: torch.device) -> str:
    """Run SFT stage only."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    datasets = get_dataset(
        config.task, tokenizer,
        sft_split=config.dataset.sft_split,
        rm_split=config.dataset.rm_split,
        ppo_split=config.dataset.ppo_split,
        seed=config.dataset.seed,
    )

    if config.task == "apps":
        sft_dataset, _ = datasets
    else:
        sft_dataset, _, _, _ = datasets

    model = SFTModel(config.model_name)
    sft_dir = train_sft(model, sft_dataset, config.sft, tokenizer, config.output_dir, device)
    return sft_dir


def train_rm_only(config: ExperimentConfig, device: torch.device, sft_checkpoint: str) -> str:
    """Run RM stage only."""
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _, rm_dataset, _, _ = get_dataset(
        config.task, tokenizer,
        sft_split=config.dataset.sft_split,
        rm_split=config.dataset.rm_split,
        ppo_split=config.dataset.ppo_split,
        seed=config.dataset.seed,
    )

    model = RewardModel(config.model_name, sft_checkpoint=sft_checkpoint)
    rm_dir = train_rm(model, rm_dataset, config.rm, config.output_dir, device)
    return rm_dir


def train_ppo_only(
    config: ExperimentConfig,
    device: torch.device,
    sft_checkpoint: str,
    rm_checkpoint: str,
    method: str = "ma_ppo",
    n_gram: int = 5,
):
    """Run PPO/MA-PPO stage only."""
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _, _, ppo_train_dataset, ppo_eval_dataset = get_dataset(
        config.task, tokenizer,
        sft_split=config.dataset.sft_split,
        rm_split=config.dataset.rm_split,
        ppo_split=config.dataset.ppo_split,
        seed=config.dataset.seed,
    )

    policy_model = PolicyModel(sft_checkpoint)
    critic_model = CriticModel(rm_checkpoint)
    reference_model = ReferenceModel(sft_checkpoint)
    reward_model = RewardModel.from_pretrained(rm_checkpoint)

    if method == "ma_ppo":
        trainer = MAPPO(
            policy_model=policy_model,
            critic_model=critic_model,
            reference_model=reference_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            clip_ratio=config.ppo.clip_ratio,
            gae_gamma=config.ppo.gae_gamma,
            gae_lambda=config.ppo.gae_lambda,
            kl_coefficient=config.ppo.kl_coefficient,
            max_prompt_length=config.ppo.max_prompt_length,
            max_response_length=config.ppo.max_response_length,
            temperature=config.ppo.temperature,
            top_p=config.ppo.top_p,
            top_k=config.ppo.top_k,
            termination=config.ma_ppo.termination,
            n_gram=n_gram,
            n_gram_list=config.ma_ppo.n_gram_list,
            n_gram_repeat_times=config.ma_ppo.n_gram_repeat_times,
            parsing_cutoff=config.ma_ppo.parsing_cutoff,
            value_estimation=config.ma_ppo.value_estimation,
        )
    else:
        trainer = VanillaPPO(
            policy_model=policy_model,
            critic_model=critic_model,
            reference_model=reference_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            clip_ratio=config.ppo.clip_ratio,
            gae_gamma=config.ppo.gae_gamma,
            gae_lambda=config.ppo.gae_lambda,
            kl_coefficient=config.ppo.kl_coefficient,
            max_prompt_length=config.ppo.max_prompt_length,
            max_response_length=config.ppo.max_response_length,
            temperature=config.ppo.temperature,
            top_p=config.ppo.top_p,
            top_k=config.ppo.top_k,
        )

    final_dir = train_rlhf(
        ppo_trainer=trainer,
        dataset=ppo_train_dataset,
        ppo_config=config.ppo,
        output_dir=config.output_dir,
        device=device,
        max_steps=5000,
        save_every=500,
        use_wandb=config.use_wandb,
        method_name=method.upper(),
    )
    return final_dir


def train_full_pipeline(config: ExperimentConfig, device: torch.device):
    """Run complete three-stage training pipeline."""
    logger.info(f"Starting full pipeline: {config.task} with {config.model_name}")
    logger.info(f"Method: {config.method}")

    # Stage 1: SFT
    sft_dir = os.path.join(config.output_dir, "sft_model")
    if not os.path.exists(sft_dir):
        sft_dir = train_sft_only(config, device)

    # Stage 2: Reward Modeling
    if config.task != "apps":
        rm_dir = os.path.join(config.output_dir, "rm_model")
        if not os.path.exists(rm_dir):
            rm_dir = train_rm_only(config, device, sft_dir)
    else:
        rm_dir = sft_dir  # Use SFT as critic initialization for APPS

    # Stage 3: PPO / MA-PPO
    train_ppo_only(
        config=config,
        device=device,
        sft_checkpoint=sft_dir,
        rm_checkpoint=rm_dir,
        method=config.method,
        n_gram=config.ma_ppo.n_gram,
    )

    logger.info("Full pipeline complete!")


def main():
    parser = argparse.ArgumentParser(description="MA-RLHF Training Pipeline")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--model", type=str, default="gemma-2b",
                        choices=["gemma-2b", "gemma-7b", "gemma-27b",
                                 "codegemma-2b", "codegemma-7b"])
    parser.add_argument("--task", type=str, default="tldr",
                        choices=["tldr", "hh-rlhf", "webgpt", "apps"])
    parser.add_argument("--method", type=str, default="ma_ppo",
                        choices=["vanilla_ppo", "ma_ppo"])
    parser.add_argument("--n_gram", type=int, default=5)
    parser.add_argument("--termination", type=str, default="ngram",
                        choices=["ngram", "randomized_ngram", "ppl", "parser"])
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--stage", type=str, default="full",
                        choices=["full", "sft", "rm", "ppo"])
    parser.add_argument("--sft_checkpoint", type=str, default=None)
    parser.add_argument("--rm_checkpoint", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load config
    if args.config:
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
        config = ExperimentConfig(**config_dict)
    else:
        config_fn = CONFIG_MAP.get(args.model)
        if config_fn is None:
            raise ValueError(f"Unknown model: {args.model}")
        config = config_fn(args.task)

    config.method = args.method
    config.output_dir = args.output_dir
    config.use_wandb = args.use_wandb
    config.dataset.seed = args.seed
    config.ma_ppo.n_gram = args.n_gram
    config.ma_ppo.termination = args.termination

    os.makedirs(config.output_dir, exist_ok=True)

    if config.use_wandb:
        wandb.init(project="ma-rlhf", config=config.__dict__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if args.stage == "full":
        train_full_pipeline(config, device)
    elif args.stage == "sft":
        train_sft_only(config, device)
    elif args.stage == "rm":
        if args.sft_checkpoint is None:
            args.sft_checkpoint = os.path.join(config.output_dir, "sft_model")
        train_rm_only(config, device, args.sft_checkpoint)
    elif args.stage == "ppo":
        if args.sft_checkpoint is None:
            args.sft_checkpoint = os.path.join(config.output_dir, "sft_model")
        if args.rm_checkpoint is None:
            args.rm_checkpoint = os.path.join(config.output_dir, "rm_model")
        train_ppo_only(
            config, device, args.sft_checkpoint, args.rm_checkpoint,
            method=args.method, n_gram=args.n_gram,
        )

    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
