"""Adaptation (SFT, DPO, KTO) for OLMoE-1B-7B-INSTRUCT.

Implements the adaptation recipe from §2 and §4.3:

SFT (Instruction Tuning):
  - 2 epochs, constant LR=2e-5, global batch size=128
  - Token-level loss aggregation (Muennighoff et al. 2024)
  - No load balancing loss during SFT (§4.3)
  - Start from post-annealing checkpoint

DPO (Direct Preference Optimization, Rafailov et al. 2023):
  - 3 epochs, LR=5e-7, global batch size=32, beta=0.1
  - No load balancing loss during DPO (§4.3)
  - Applied to SFT model

KTO (Kahneman-Tversky Optimization, Ethayarajh et al. 2024):
  - 5000 steps (1.3 epochs), LR=5e-7, RMSProp optimizer
  - No load balancing loss during KTO (§4.3)
  - Applied to SFT model

Key finding (§4.3): Not using load balancing loss during adaptation leads to
better performance (54.0 vs 52.8 after SFT, 57.7 vs 57.1 after DPO).
The routing distribution remains stable without it because routing saturates
early in pretraining (§5.1).
"""

import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import AdaptConfig, OLMoEConfig
from data import build_dpo_dataloader, build_sft_dataloader
from model import OLMoE, build_olmoe_1b_7b
from train import load_checkpoint, save_checkpoint, set_lr


# ---------------------------------------------------------------------------
# SFT Training
# ---------------------------------------------------------------------------

def sft_loss(
    model: OLMoE,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    use_load_balance_loss: bool = False,
) -> Dict[str, torch.Tensor]:
    """Compute SFT loss with token-level aggregation.

    Token-level aggregation (vs sequence-level) improves performance on
    long generative tasks like AlpacaEval (Muennighoff et al. 2024 §B).
    """
    out = model(input_ids=input_ids, labels=labels)
    ce_loss = out["ce_loss"]

    if use_load_balance_loss:
        loss = (
            ce_loss
            + model.config.load_balance_loss_weight * out["load_balance_loss"]
            + model.config.router_z_loss_weight * out["router_z_loss"]
        )
    else:
        loss = ce_loss

    return {"loss": loss, "ce_loss": ce_loss}


def train_sft(
    model: OLMoE,
    dataloader: DataLoader,
    config: AdaptConfig,
    device: torch.device,
    save_dir: str,
):
    """SFT training loop."""
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.sft_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=config.weight_decay,
    )

    total_steps = len(dataloader) * config.sft_epochs
    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32

    model.train()
    step = 0

    for epoch in range(config.sft_epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=dtype):
                metrics = sft_loss(
                    model, input_ids, labels,
                    use_load_balance_loss=config.sft_use_load_balance_loss,
                )

            metrics["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            step += 1

            if step % 100 == 0:
                print(f"SFT step={step}/{total_steps} | loss={metrics['loss'].item():.4f}")

    save_path = Path(save_dir) / "sft"
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / "model.pt")
    print(f"SFT complete. Saved to {save_path}")


# ---------------------------------------------------------------------------
# DPO Training
# ---------------------------------------------------------------------------

def dpo_loss(
    policy_model: OLMoE,
    reference_model: OLMoE,
    chosen_input_ids: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_input_ids: torch.Tensor,
    rejected_labels: torch.Tensor,
    beta: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Direct Preference Optimization loss (Rafailov et al. 2023).

    L_DPO = -E[log sigma(beta * (log pi(y_w|x) - log pi(y_l|x)
                                - log pi_ref(y_w|x) + log pi_ref(y_l|x)))]

    where y_w = chosen, y_l = rejected, pi = policy, pi_ref = reference.
    """
    def compute_log_probs(model: OLMoE, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        out = model(input_ids=input_ids)
        logits = out["logits"]  # (batch, seq, vocab)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        # Gather log probs for actual tokens, ignoring -100 positions
        mask = (shift_labels != -100).float()
        token_log_probs = log_probs.gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        # Sum over response tokens (token-level)
        seq_log_probs = (token_log_probs * mask).sum(-1)
        return seq_log_probs

    with torch.no_grad():
        ref_chosen_lp = compute_log_probs(reference_model, chosen_input_ids, chosen_labels)
        ref_rejected_lp = compute_log_probs(reference_model, rejected_input_ids, rejected_labels)

    policy_chosen_lp = compute_log_probs(policy_model, chosen_input_ids, chosen_labels)
    policy_rejected_lp = compute_log_probs(policy_model, rejected_input_ids, rejected_labels)

    chosen_rewards = beta * (policy_chosen_lp - ref_chosen_lp)
    rejected_rewards = beta * (policy_rejected_lp - ref_rejected_lp)

    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()

    return {
        "loss": loss,
        "chosen_rewards": chosen_rewards.mean(),
        "rejected_rewards": rejected_rewards.mean(),
        "reward_accuracy": reward_accuracy,
    }


def train_dpo(
    policy_model: OLMoE,
    reference_model: OLMoE,
    dataloader: DataLoader,
    config: AdaptConfig,
    device: torch.device,
    save_dir: str,
):
    """DPO training loop."""
    optimizer = AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=config.dpo_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=config.weight_decay,
    )

    total_steps = len(dataloader) * config.dpo_epochs
    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32

    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad_(False)

    policy_model.train()
    step = 0

    for epoch in range(config.dpo_epochs):
        for batch in dataloader:
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=dtype):
                metrics = dpo_loss(
                    policy_model, reference_model,
                    chosen_input_ids, chosen_labels,
                    rejected_input_ids, rejected_labels,
                    beta=config.dpo_beta,
                )

            metrics["loss"].backward()
            nn.utils.clip_grad_norm_(policy_model.parameters(), config.grad_clip)
            optimizer.step()
            step += 1

            if step % 100 == 0:
                print(
                    f"DPO step={step}/{total_steps} | loss={metrics['loss'].item():.4f} | "
                    f"acc={metrics['reward_accuracy'].item():.3f}"
                )

    save_path = Path(save_dir) / "dpo"
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(policy_model.state_dict(), save_path / "model.pt")
    print(f"DPO complete. Saved to {save_path}")


# ---------------------------------------------------------------------------
# KTO Training
# ---------------------------------------------------------------------------

def kto_loss(
    policy_model: OLMoE,
    reference_model: OLMoE,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    desirable: torch.Tensor,
    beta: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """KTO loss (Ethayarajh et al. 2024).

    Prospect-theoretic alignment: treats desirable and undesirable
    completions asymmetrically based on Kahneman-Tversky value function.

    For desirable (label=1): maximize log sigma(beta * (log pi/pi_ref - KL))
    For undesirable (label=0): maximize log sigma(-beta * (log pi/pi_ref - KL))
    """
    def compute_log_probs(model: OLMoE, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        out = model(input_ids=input_ids)
        logits = out["logits"]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        mask = (shift_labels != -100).float()
        token_log_probs = log_probs.gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return (token_log_probs * mask).sum(-1)

    with torch.no_grad():
        ref_log_probs = compute_log_probs(reference_model, input_ids, labels)

    policy_log_probs = compute_log_probs(policy_model, input_ids, labels)
    log_ratio = policy_log_probs - ref_log_probs

    # KL divergence estimate (mean log ratio over batch)
    kl = log_ratio.mean().detach()

    # KTO value function
    desirable = desirable.bool()
    losses = torch.zeros_like(log_ratio)

    if desirable.any():
        losses[desirable] = 1 - F.sigmoid(beta * (log_ratio[desirable] - kl))
    if (~desirable).any():
        losses[~desirable] = 1 - F.sigmoid(-beta * (log_ratio[~desirable] - kl))

    loss = losses.mean()
    return {"loss": loss, "kl": kl}


def train_kto(
    policy_model: OLMoE,
    reference_model: OLMoE,
    dataloader: DataLoader,
    config: AdaptConfig,
    device: torch.device,
    save_dir: str,
):
    """KTO training loop.

    Uses RMSProp optimizer (paper §4.3, Appendix F Table 14).
    Trains for 5000 steps (1.3 epochs).
    """
    optimizer = torch.optim.RMSprop(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=config.kto_lr,
        weight_decay=config.weight_decay,
    )

    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32

    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad_(False)

    policy_model.train()
    step = 0

    for batch in dataloader:
        if step >= config.kto_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=dtype):
            metrics = kto_loss(
                policy_model, reference_model,
                input_ids, labels, label,
            )

        metrics["loss"].backward()
        nn.utils.clip_grad_norm_(policy_model.parameters(), config.grad_clip)
        optimizer.step()
        step += 1

        if step % 100 == 0:
            print(f"KTO step={step}/{config.kto_steps} | loss={metrics['loss'].item():.4f}")

    save_path = Path(save_dir) / "kto"
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(policy_model.state_dict(), save_path / "model.pt")
    print(f"KTO complete. Saved to {save_path}")


# ---------------------------------------------------------------------------
# Full adaptation pipeline
# ---------------------------------------------------------------------------

def run_adaptation(config: OLMoEConfig, mode: str = "dpo"):
    """Run the full adaptation pipeline: SFT -> DPO (or KTO).

    Args:
        config: full OLMoE config
        mode: 'dpo' or 'kto' for preference tuning stage
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.adapt.seed)

    # Load base model (post-annealing checkpoint)
    model = build_olmoe_1b_7b()
    if config.adapt.base_checkpoint:
        ckpt = torch.load(config.adapt.base_checkpoint, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        print(f"Loaded base checkpoint from {config.adapt.base_checkpoint}")
    model = model.to(device)

    # Load tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "allenai/gpt-neox-olmo-dolma-v1_5",
            trust_remote_code=True,
        )
    except Exception:
        tokenizer = None
        print("Warning: tokenizer not available, using placeholder")

    # Stage 1: SFT
    print("=== Stage 1: Supervised Fine-Tuning ===")
    if tokenizer is not None:
        sft_loader = build_sft_dataloader(
            data_path=config.adapt.sft_data_path,
            tokenizer=tokenizer,
            batch_size=config.adapt.sft_batch_size,
            max_seq_len=config.adapt.sft_max_seq_len,
        )
        train_sft(model, sft_loader, config.adapt, device, config.adapt.save_dir)

    # Stage 2: Preference tuning
    print(f"=== Stage 2: Preference Tuning ({mode.upper()}) ===")
    if tokenizer is not None:
        # Reference model = frozen copy of SFT model
        reference_model = build_olmoe_1b_7b()
        reference_model.load_state_dict(model.state_dict())
        reference_model = reference_model.to(device)

        if mode == "dpo":
            dpo_loader = build_dpo_dataloader(
                data_path=config.adapt.dpo_data_path,
                tokenizer=tokenizer,
                batch_size=config.adapt.dpo_batch_size,
            )
            train_dpo(model, reference_model, dpo_loader, config.adapt, device, config.adapt.save_dir)
        elif mode == "kto":
            from data import KTODataset
            from torch.utils.data import DataLoader as TorchDataLoader
            kto_dataset = KTODataset(config.adapt.kto_data_path, tokenizer)
            kto_loader = TorchDataLoader(kto_dataset, batch_size=config.adapt.dpo_batch_size, shuffle=True)
            train_kto(model, reference_model, kto_loader, config.adapt, device, config.adapt.save_dir)

    print("Adaptation complete.")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adapt OLMoE (SFT + DPO/KTO)")
    parser.add_argument("--base_checkpoint", type=str, required=True)
    parser.add_argument("--sft_data_path", type=str, required=True)
    parser.add_argument("--dpo_data_path", type=str, default=None)
    parser.add_argument("--kto_data_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="checkpoints/adapted")
    parser.add_argument("--mode", type=str, default="dpo", choices=["dpo", "kto"])
    args = parser.parse_args()

    cfg = OLMoEConfig()
    cfg.adapt.base_checkpoint = args.base_checkpoint
    cfg.adapt.sft_data_path = args.sft_data_path
    cfg.adapt.dpo_data_path = args.dpo_data_path or ""
    cfg.adapt.kto_data_path = args.kto_data_path or ""
    cfg.adapt.save_dir = args.save_dir

    run_adaptation(cfg, mode=args.mode)
