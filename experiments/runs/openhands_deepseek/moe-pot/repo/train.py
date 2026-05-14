import os
import sys
import argparse
import math
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    ModelConfig, TrainingConfig, MODEL_CONFIGS,
    PRETRAIN_DATASETS, DATASET_SIZES,
)
from model import MoEPOT
from data import (
    PDEDataset, create_multi_dataset_loaders,
    create_single_dataset_loader, DATASET_INFO,
)
from modules import Block


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def l2_relative_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute L2 relative error.

    L2RE = ||pred - target||_2 / ||target||_2
    """
    with torch.no_grad():
        diff = pred - target
        num = torch.norm(diff, p=2)
        den = torch.norm(target, p=2)
        if den < 1e-8:
            return float('nan')
        return (num / den).item()


# ---------------------------------------------------------------------------
# One-cycle learning rate scheduler
# ---------------------------------------------------------------------------

class OneCycleLR:
    """One-cycle learning rate schedule.

    Linear warmup from lr/10 to lr_max, then cosine decay to lr/1000.
    """

    def __init__(self, optimizer: optim.Optimizer, lr_max: float,
                 total_epochs: int, warmup_epochs: int):
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.lr_min = lr_max / 1000.0
        self.lr_start = lr_max / 10.0

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            # Linear warmup
            progress = epoch / max(1, self.warmup_epochs)
            lr = self.lr_start + (self.lr_max - self.lr_start) * progress
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1.0 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def create_model(model_size: str, in_channels: int, out_channels: int,
                 device: torch.device) -> MoEPOT:
    """Create MoE-POT model of specified size."""
    cfg = MODEL_CONFIGS[model_size]
    model = MoEPOT(
        in_channels=in_channels,
        out_channels=out_channels,
        dim=cfg.attention_dim,
        mlp_dim=cfg.mlp_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        num_routed_experts=cfg.num_routed_experts,
        num_shared_experts=cfg.num_shared_experts,
        top_k=cfg.top_k,
        patch_size=cfg.patch_size,
        fourier_modes=cfg.fourier_modes,
        spatial_resolution=cfg.spatial_resolution,
        expert_kernel_size=cfg.expert_kernel_size,
    )
    return model.to(device)


# ---------------------------------------------------------------------------
# Pre-training
# ---------------------------------------------------------------------------

def pretrain(model: MoEPOT, train_config: TrainingConfig,
             device: torch.device, data_root: str = "./data"):
    """Pre-train MoE-POT on multiple PDE datasets."""
    print("=" * 60)
    print("Starting Pre-training")
    print("=" * 60)

    model.set_num_timesteps(train_config.num_timesteps_in)
    model.train()

    # Data loader with balanced sampling
    train_loader = create_multi_dataset_loaders(
        dataset_names=PRETRAIN_DATASETS,
        split="train",
        target_resolution=128,
        max_channels=8,
        num_timesteps_in=train_config.num_timesteps_in,
        batch_size=train_config.pretrain_batch_size,
        add_noise=True,
        noise_eps=train_config.noise_eps,
        dataset_weights=train_config.dataset_weights,
        data_root=data_root,
    )

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config.pretrain_lr,
        weight_decay=train_config.weight_decay,
        betas=(train_config.beta1, train_config.beta2),
    )

    # Scheduler
    scheduler = OneCycleLR(
        optimizer,
        lr_max=train_config.pretrain_lr,
        total_epochs=train_config.pretrain_epochs,
        warmup_epochs=train_config.pretrain_warmup_epochs,
    )

    # Training loop
    for epoch in range(train_config.pretrain_epochs):
        lr = scheduler.step(epoch)
        epoch_loss = 0.0
        epoch_lb_loss = 0.0
        epoch_pred_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_config.pretrain_epochs}")
        for batch in pbar:
            u_in = batch["input"].to(device)   # (B, C, H, W, T)
            u_target = batch["target"].to(device)  # (B, C, H, W)

            optimizer.zero_grad()

            pred, lb_loss = model(u_in)

            pred_loss = nn.functional.mse_loss(pred, u_target)
            loss = pred_loss + lb_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_lb_loss += lb_loss.item()
            epoch_pred_loss += pred_loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "pred": f"{pred_loss.item():.6f}",
                "lb": f"{lb_loss.item():.6f}",
                "lr": f"{lr:.6f}",
            })

        avg_loss = epoch_loss / len(train_loader)
        avg_lb = epoch_lb_loss / len(train_loader)
        avg_pred = epoch_pred_loss / len(train_loader)

        print(f"Epoch {epoch+1}: loss={avg_loss:.6f}, "
              f"pred_loss={avg_pred:.6f}, lb_loss={avg_lb:.6f}, lr={lr:.6f}")

        # Save checkpoint periodically
        if (epoch + 1) % 100 == 0 or epoch == 0:
            save_path = os.path.join(
                train_config.save_dir, f"pretrain_epoch{epoch+1}.pt")
            os.makedirs(train_config.save_dir, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, save_path)
            print(f"Checkpoint saved: {save_path}")

    # Save final model
    final_path = os.path.join(train_config.save_dir, "pretrain_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Pre-training complete. Model saved to {final_path}")


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def finetune(model: MoEPOT, dataset_name: str,
             train_config: TrainingConfig,
             device: torch.device, data_root: str = "./data"):
    """Fine-tune MoE-POT on a single PDE dataset with frozen router."""
    print("=" * 60)
    print(f"Fine-tuning on {dataset_name}")
    print("=" * 60)

    model.set_num_timesteps(train_config.num_timesteps_in)

    # Freeze router-gating network parameters
    for module in model.modules():
        if isinstance(module, Block):
            # Freeze router
            for p in module.moe.router.parameters():
                p.requires_grad = False

    model.train()

    train_loader = create_single_dataset_loader(
        dataset_name=dataset_name,
        split="train",
        target_resolution=128,
        max_channels=8,
        num_timesteps_in=train_config.num_timesteps_in,
        batch_size=train_config.pretrain_batch_size,
        add_noise=False,
        data_root=data_root,
    )

    test_loader = create_single_dataset_loader(
        dataset_name=dataset_name,
        split="test",
        target_resolution=128,
        max_channels=8,
        num_timesteps_in=train_config.num_timesteps_in,
        batch_size=train_config.pretrain_batch_size,
        add_noise=False,
        data_root=data_root,
    )

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_config.finetune_lr,
        weight_decay=train_config.weight_decay,
        betas=(train_config.beta1, train_config.beta2),
    )

    scheduler = OneCycleLR(
        optimizer,
        lr_max=train_config.finetune_lr,
        total_epochs=train_config.finetune_epochs,
        warmup_epochs=train_config.finetune_warmup_epochs,
    )

    best_l2re = float('inf')

    for epoch in range(train_config.finetune_epochs):
        lr = scheduler.step(epoch)
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"FT Epoch {epoch+1}/{train_config.finetune_epochs}")
        for batch in pbar:
            u_in = batch["input"].to(device)
            u_target = batch["target"].to(device)

            optimizer.zero_grad()
            pred, lb_loss = model(u_in)
            loss = nn.functional.mse_loss(pred, u_target) + lb_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}", "lr": f"{lr:.6f}"})

        avg_loss = epoch_loss / len(train_loader)
        print(f"FT Epoch {epoch+1}: loss={avg_loss:.6f}, lr={lr:.6f}")

        # Evaluate every 20 epochs
        if (epoch + 1) % 20 == 0:
            l2re = evaluate_zero_shot(model, test_loader, device)
            print(f"  Test L2RE: {l2re:.6f}")
            if l2re < best_l2re:
                best_l2re = l2re
                best_path = os.path.join(
                    train_config.save_dir, f"finetune_{dataset_name}_best.pt")
                torch.save(model.state_dict(), best_path)

    final_path = os.path.join(
        train_config.save_dir, f"finetune_{dataset_name}_final.pt")
    torch.save(model.state_dict(), final_path)

    # Final evaluation
    final_l2re = evaluate_zero_shot(model, test_loader, device)
    print(f"Fine-tuning complete. Final L2RE: {final_l2re:.6f}")
    return final_l2re


# ---------------------------------------------------------------------------
# Downstream task fine-tuning
# ---------------------------------------------------------------------------

def downstream_finetune(model: MoEPOT, dataset_name: str,
                         train_config: TrainingConfig,
                         device: torch.device, data_root: str = "./data"):
    """Fine-tune on downstream tasks (500 epochs)."""
    print("=" * 60)
    print(f"Downstream Fine-tuning on {dataset_name}")
    print("=" * 60)

    model.set_num_timesteps(train_config.num_timesteps_in)

    # Freeze router
    for module in model.modules():
        if isinstance(module, Block):
            for p in module.moe.router.parameters():
                p.requires_grad = False

    model.train()

    train_loader = create_single_dataset_loader(
        dataset_name=dataset_name,
        split="train",
        target_resolution=128,
        max_channels=8,
        num_timesteps_in=train_config.num_timesteps_in,
        batch_size=train_config.pretrain_batch_size,
        add_noise=False,
        data_root=data_root,
    )

    test_loader = create_single_dataset_loader(
        dataset_name=dataset_name,
        split="test",
        target_resolution=128,
        max_channels=8,
        num_timesteps_in=train_config.num_timesteps_in,
        batch_size=train_config.pretrain_batch_size,
        add_noise=False,
        data_root=data_root,
    )

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_config.downstream_lr,
        weight_decay=train_config.weight_decay,
        betas=(train_config.beta1, train_config.beta2),
    )

    scheduler = OneCycleLR(
        optimizer,
        lr_max=train_config.downstream_lr,
        total_epochs=train_config.downstream_epochs,
        warmup_epochs=train_config.downstream_warmup_epochs,
    )

    best_l2re = float('inf')

    for epoch in range(train_config.downstream_epochs):
        lr = scheduler.step(epoch)
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"DS Epoch {epoch+1}/"
                    f"{train_config.downstream_epochs}")
        for batch in pbar:
            u_in = batch["input"].to(device)
            u_target = batch["target"].to(device)

            optimizer.zero_grad()
            pred, lb_loss = model(u_in)
            loss = nn.functional.mse_loss(pred, u_target) + lb_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 50 == 0:
            print(f"DS Epoch {epoch+1}: loss={avg_loss:.6f}, lr={lr:.6f}")

        if (epoch + 1) % 50 == 0:
            l2re = evaluate_zero_shot(model, test_loader, device)
            print(f"  Test L2RE: {l2re:.6f}")
            if l2re < best_l2re:
                best_l2re = l2re
                best_path = os.path.join(
                    train_config.save_dir,
                    f"downstream_{dataset_name}_best.pt")
                torch.save(model.state_dict(), best_path)

    final_l2re = evaluate_zero_shot(model, test_loader, device)
    print(f"Downstream fine-tuning complete. Final L2RE: {final_l2re:.6f}")
    return final_l2re


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_zero_shot(model: MoEPOT, test_loader: DataLoader,
                        device: torch.device) -> float:
    """Evaluate zero-shot L2 relative error on a test set."""
    model.eval()
    total_l2re = 0.0
    count = 0

    with torch.no_grad():
        for batch in test_loader:
            u_in = batch["input"].to(device)
            u_target = batch["target"].to(device)
            pred, _ = model(u_in)
            for b in range(u_target.size(0)):
                l2re_val = l2_relative_error(pred[b], u_target[b])
                if not math.isnan(l2re_val):
                    total_l2re += l2re_val
                    count += 1

    model.train()
    return total_l2re / max(1, count)


def evaluate_rollout(model: MoEPOT, test_loader: DataLoader,
                      device: torch.device, rollout_steps: int = 100) -> List[float]:
    """Evaluate auto-regressive rollout error over multiple steps.

    Returns list of L2RE per rollout step.
    """
    model.eval()
    step_errors = defaultdict(list)

    with torch.no_grad():
        for batch in test_loader:
            u_in = batch["input"].to(device)
            u_target_all = batch["target"].to(device)

            preds = model.autoregressive_rollout(u_in, rollout_steps)
            for step, pred in enumerate(preds):
                l2re_val = l2_relative_error(pred, u_target_all)
                if not math.isnan(l2re_val):
                    step_errors[step].append(l2re_val)

    model.train()

    avg_errors = [np.mean(step_errors[s]) for s in sorted(step_errors.keys())]
    return avg_errors


def evaluate_all_datasets(model: MoEPOT, dataset_names: List[str],
                           train_config: TrainingConfig,
                           device: torch.device,
                           data_root: str = "./data") -> Dict[str, float]:
    """Evaluate zero-shot L2RE on multiple datasets."""
    results = {}
    for name in dataset_names:
        loader = create_single_dataset_loader(
            dataset_name=name,
            split="test",
            target_resolution=128,
            max_channels=8,
            num_timesteps_in=train_config.num_timesteps_in,
            batch_size=train_config.pretrain_batch_size,
            add_noise=False,
            data_root=data_root,
        )
        l2re = evaluate_zero_shot(model, loader, device)
        results[name] = l2re
        print(f"  {name}: L2RE = {l2re:.6f}")
    return results


# ---------------------------------------------------------------------------
# Interpretability Analysis
# ---------------------------------------------------------------------------

def collect_router_statistics(model: MoEPOT, dataset_name: str,
                               train_config: TrainingConfig,
                               device: torch.device,
                               data_root: str = "./data",
                               block_idx: int = 1) -> torch.Tensor:
    """Collect average router-gating network outputs for a dataset."""
    model.eval()

    loader = create_single_dataset_loader(
        dataset_name=dataset_name,
        split="test",
        target_resolution=128,
        max_channels=8,
        num_timesteps_in=train_config.num_timesteps_in,
        batch_size=train_config.pretrain_batch_size,
        add_noise=False,
        data_root=data_root,
    )

    all_logits = []
    with torch.no_grad():
        for batch in loader:
            u_in = batch["input"].to(device)
            # Forward through model up to the target block's router
            z_p = model.patchify(u_in)
            z = model.temporal_agg(z_p)
            for i, block in enumerate(model.blocks):
                if i == block_idx:
                    # Get router logits for this block
                    identity = z
                    z0 = z.permute(0, 2, 3, 1)
                    z0 = block.norm1(z0)
                    z0 = z0.permute(0, 3, 1, 2)
                    z0 = block.fourier(z0)
                    z0 = z0 + identity
                    logits = block.moe.router(z0)
                    all_logits.append(logits.cpu())
                z, _ = block(z)

    if len(all_logits) == 0:
        return torch.zeros(1, model.blocks[0].moe.num_routed)

    all_logits = torch.cat(all_logits, dim=0)  # (N, num_routed)
    avg_logits = all_logits.mean(dim=0)  # (num_routed,)
    return avg_logits


def classify_by_router(model: MoEPOT,
                        dataset_names: List[str],
                        train_config: TrainingConfig,
                        device: torch.device,
                        data_root: str = "./data",
                        block_idx: int = 1) -> float:
    """Classify input data by dataset using router-gating network decisions.

    Following Section 5.4:
    1. Compute average expert selection Y_i for each dataset
    2. For each test sample, compute distance to each Y_i
    3. Predict dataset with minimum cross-entropy distance
    """
    model.eval()

    # Step 1: Compute reference distributions Y_i for each dataset
    ref_distributions = {}
    for name in dataset_names:
        avg_logits = collect_router_statistics(
            model, name, train_config, device, data_root, block_idx)
        ref_distributions[name] = torch.softmax(avg_logits, dim=-1)  # (N_r,)

    # Step 2 & 3: Classify test samples
    correct = 0
    total = 0

    for name in dataset_names:
        loader = create_single_dataset_loader(
            dataset_name=name,
            split="test",
            target_resolution=128,
            max_channels=8,
            num_timesteps_in=train_config.num_timesteps_in,
            batch_size=train_config.pretrain_batch_size,
            add_noise=False,
            data_root=data_root,
        )

        for batch in loader:
            u_in = batch["input"].to(device)
            with torch.no_grad():
                z_p = model.patchify(u_in)
                z = model.temporal_agg(z_p)
                for i, block in enumerate(model.blocks):
                    if i == block_idx:
                        identity = z
                        z0 = z.permute(0, 2, 3, 1)
                        z0 = block.norm1(z0)
                        z0 = z0.permute(0, 3, 1, 2)
                        z0 = block.fourier(z0)
                        z0 = z0 + identity
                        logits = block.moe.router(z0)
                        probs = torch.softmax(logits, dim=-1)
                    z, _ = block(z)

            for b in range(probs.size(0)):
                sample_prob = probs[b]  # (N_r,)
                # Cross-entropy distance
                best_name = None
                best_dist = float('inf')
                for ref_name, ref_prob in ref_distributions.items():
                    dist = -(sample_prob * torch.log(ref_prob + 1e-8)).sum().item()
                    if dist < best_dist:
                        best_dist = dist
                        best_name = ref_name
                if best_name == name:
                    correct += 1
                total += 1

    accuracy = correct / max(1, total)
    return accuracy


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MoE-POT Training")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["pretrain", "finetune", "downstream",
                                  "evaluate", "interpret"],
                        help="Training mode")
    parser.add_argument("--model_size", type=str, default="Tiny",
                        choices=["Tiny", "Small", "Medium"],
                        help="Model size variant")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name for fine-tuning/evaluation")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for data")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pre-trained checkpoint")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--block_idx", type=int, default=1,
                        help="Block index for interpretability analysis")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_config = TrainingConfig()

    # Max channels across all datasets
    max_channels = 8
    in_channels = max_channels
    out_channels = max_channels

    if args.mode == "pretrain":
        # Pre-train from scratch
        model = create_model(args.model_size, in_channels, out_channels, device)
        pretrain(model, train_config, device, data_root=args.data_root)

    elif args.mode == "finetune":
        if args.dataset is None:
            raise ValueError("--dataset required for fine-tuning")
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for fine-tuning")

        model = create_model(args.model_size, in_channels, out_channels, device)
        state_dict = torch.load(args.checkpoint, map_location=device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)

        finetune(model, args.dataset, train_config, device,
                 data_root=args.data_root)

    elif args.mode == "downstream":
        if args.dataset is None:
            raise ValueError("--dataset required for downstream")
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for downstream")

        model = create_model(args.model_size, in_channels, out_channels, device)
        state_dict = torch.load(args.checkpoint, map_location=device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)

        downstream_finetune(model, args.dataset, train_config, device,
                            data_root=args.data_root)

    elif args.mode == "evaluate":
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for evaluation")

        model = create_model(args.model_size, in_channels, out_channels, device)
        state_dict = torch.load(args.checkpoint, map_location=device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)
        model.set_num_timesteps(train_config.num_timesteps_in)

        if args.dataset:
            results = evaluate_all_datasets(
                model, [args.dataset], train_config, device,
                data_root=args.data_root)
        else:
            results = evaluate_all_datasets(
                model, PRETRAIN_DATASETS, train_config, device,
                data_root=args.data_root)

        print("\nZero-shot Results:")
        for name, l2re in results.items():
            print(f"  {name}: {l2re:.6f}")

    elif args.mode == "interpret":
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for interpretability")

        model = create_model(args.model_size, in_channels, out_channels, device)
        state_dict = torch.load(args.checkpoint, map_location=device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)
        model.set_num_timesteps(train_config.num_timesteps_in)

        accuracy = classify_by_router(
            model, PRETRAIN_DATASETS, train_config, device,
            data_root=args.data_root, block_idx=args.block_idx)
        print(f"\nRouter classification accuracy (Block {args.block_idx}): "
              f"{accuracy * 100:.1f}%")


if __name__ == "__main__":
    main()
