"""Training script for MoE-POT.

Implements:
- Pre-training with auto-regressive denoising (Section 2.2, 4)
- Fine-tuning with frozen router (Appendix B.3)
- Downstream task fine-tuning (Section 5.2)
- OneCycleLR schedule with warmup
- Multi-GPU training via DistributedDataParallel
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import (
    DownstreamConfig,
    FinetuneConfig,
    ModelConfig,
    TrainConfig,
    get_model_config,
)
from data import (
    build_dataloader,
    build_pretrain_datasets,
    build_single_dataset,
    inject_noise,
)
from model import MoEPOT, build_model
from utils import AverageMeter, compute_l2_relative_error, save_checkpoint, load_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pretrain(
    model_size: str,
    data_root: str,
    train_cfg: TrainConfig,
    resume: Optional[str] = None,
    local_rank: int = 0,
    world_size: int = 1,
) -> None:
    """Pre-train MoE-POT on mixed PDE datasets.

    Loss = Σ_t ||G_w(u^{<t} + ε) - u^t||_2^2 + Σ_l L_balance^l
    """
    set_seed(train_cfg.seed + local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Build model
    model = build_model(model_size).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    raw_model = model.module if world_size > 1 else model

    # Build dataset
    train_dataset = build_pretrain_datasets(
        data_root=data_root,
        split="train",
        num_input_frames=train_cfg.num_timesteps,
    )

    if world_size > 1:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg.batch_size // world_size,
            sampler=sampler,
            num_workers=train_cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = build_dataloader(
            train_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=train_cfg.num_workers,
            use_weighted_sampler=True,
        )

    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=train_cfg.lr,
        betas=(train_cfg.beta1, train_cfg.beta2),
        weight_decay=train_cfg.weight_decay,
    )

    # OneCycleLR: total steps = num_epochs * steps_per_epoch
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=train_cfg.lr,
        total_steps=train_cfg.num_epochs * steps_per_epoch,
        pct_start=train_cfg.warmup_epochs / train_cfg.num_epochs,
        anneal_strategy="cos",
    )

    start_epoch = 0
    if resume:
        start_epoch = load_checkpoint(resume, raw_model, optimizer, scheduler)

    checkpoint_dir = Path(train_cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if local_rank == 0:
        print(f"Model parameters: {raw_model.count_parameters()}")
        print(f"Training for {train_cfg.num_epochs} epochs on {len(train_dataset)} samples")

    for epoch in range(start_epoch, train_cfg.num_epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)

        model.train()
        loss_meter = AverageMeter()
        pred_loss_meter = AverageMeter()
        bal_loss_meter = AverageMeter()

        t0 = time.time()
        for batch in train_loader:
            inputs = batch["input"].to(device)    # (B, T, C, H, W)
            targets = batch["target"].to(device)  # (B, C, H, W)

            # Inject noise into input frames (pre-training only)
            noisy_inputs = inject_noise(inputs, noise_scale=train_cfg.noise_scale)

            optimizer.zero_grad()

            # Auto-regressive prediction loss: Σ_t ||G(u^{<t} + ε) - u^t||^2
            pred_loss = torch.tensor(0.0, device=device)
            total_balance_loss = torch.tensor(0.0, device=device)

            T = inputs.shape[1]
            for t in range(1, T + 1):
                # Use frames 0..t-1 as input, predict frame t
                # For t < T: target is inputs[:, t]; for t == T: target is targets
                if t < T:
                    target_t = inputs[:, t]
                else:
                    target_t = targets

                input_window = noisy_inputs[:, :t]  # (B, t, C, H, W)

                # Pad to T frames if needed (repeat last frame)
                if t < T:
                    pad_len = T - t
                    pad = input_window[:, -1:].expand(-1, pad_len, -1, -1, -1)
                    input_window = torch.cat([input_window, pad], dim=1)

                pred, balance_loss = model(input_window)
                step_loss = F.mse_loss(pred, target_t)
                pred_loss = pred_loss + step_loss
                total_balance_loss = total_balance_loss + balance_loss

            loss = pred_loss + total_balance_loss
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            B = inputs.shape[0]
            loss_meter.update(loss.item(), B)
            pred_loss_meter.update(pred_loss.item(), B)
            bal_loss_meter.update(total_balance_loss.item(), B)

        if local_rank == 0:
            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch+1}/{train_cfg.num_epochs}] "
                f"Loss: {loss_meter.avg:.4f} "
                f"(pred: {pred_loss_meter.avg:.4f}, bal: {bal_loss_meter.avg:.4f}) "
                f"LR: {scheduler.get_last_lr()[0]:.2e} "
                f"Time: {elapsed:.1f}s"
            )

            if (epoch + 1) % train_cfg.save_every == 0 or epoch == train_cfg.num_epochs - 1:
                save_checkpoint(
                    checkpoint_dir / f"pretrain_{model_size}_epoch{epoch+1}.pt",
                    raw_model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                )


def finetune(
    model_size: str,
    data_root: str,
    dataset_name: str,
    pretrain_ckpt: str,
    ft_cfg: FinetuneConfig,
    local_rank: int = 0,
    world_size: int = 1,
) -> None:
    """Fine-tune a pre-trained MoE-POT on a single dataset.

    Router-gating network is frozen; only expert networks are updated.
    """
    set_seed(42 + local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model = build_model(model_size).to(device)
    load_checkpoint(pretrain_ckpt, model)

    if ft_cfg.freeze_router:
        model.freeze_router()
        if local_rank == 0:
            print("Router-gating network frozen for fine-tuning.")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    raw_model = model.module if world_size > 1 else model

    train_dataset = build_single_dataset(data_root, dataset_name, split="train")
    test_dataset = build_single_dataset(data_root, dataset_name, split="test")

    train_loader = build_dataloader(
        train_dataset, ft_cfg.batch_size, shuffle=True, num_workers=ft_cfg.num_workers
    )
    test_loader = build_dataloader(
        test_dataset, ft_cfg.batch_size, shuffle=False, num_workers=ft_cfg.num_workers
    )

    # Only optimize non-frozen parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(
        trainable_params,
        lr=ft_cfg.lr,
        betas=(ft_cfg.beta1, ft_cfg.beta2),
        weight_decay=ft_cfg.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=ft_cfg.lr,
        total_steps=ft_cfg.num_epochs * steps_per_epoch,
        pct_start=ft_cfg.warmup_epochs / ft_cfg.num_epochs,
        anneal_strategy="cos",
    )

    checkpoint_dir = Path(ft_cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(ft_cfg.num_epochs):
        model.train()
        loss_meter = AverageMeter()

        for batch in train_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            pred, balance_loss = model(inputs)
            loss = F.mse_loss(pred, targets) + balance_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            loss_meter.update(loss.item(), inputs.shape[0])

        if local_rank == 0 and (epoch + 1) % 20 == 0:
            l2re = evaluate_model(model, test_loader, device)
            print(
                f"[FT {dataset_name}] Epoch [{epoch+1}/{ft_cfg.num_epochs}] "
                f"Loss: {loss_meter.avg:.4f}  L2RE: {l2re:.4f}"
            )

    if local_rank == 0:
        save_checkpoint(
            checkpoint_dir / f"finetune_{model_size}_{dataset_name}.pt",
            raw_model,
        )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model on a DataLoader and return mean L2 relative error."""
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            pred, _ = model(inputs)
            err = compute_l2_relative_error(pred, targets)
            errors.append(err.item())
    return float(np.mean(errors))


def evaluate_rollout(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_rollout_steps: int = 10,
) -> Dict[int, float]:
    """Evaluate auto-regressive rollout error at multiple timesteps.

    Returns a dict mapping step index → mean L2RE.
    """
    model.eval()
    step_errors: Dict[int, List[float]] = {s: [] for s in range(1, num_rollout_steps + 1)}

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)   # (B, T, C, H, W)
            targets = batch["target"].to(device)  # (B, C, H, W)

            B, T, C, H, W = inputs.shape
            current_window = inputs.clone()

            for step in range(1, num_rollout_steps + 1):
                pred, _ = model(current_window)
                if step == num_rollout_steps:
                    err = compute_l2_relative_error(pred, targets)
                    step_errors[step].append(err.item())
                else:
                    step_errors[step].append(
                        compute_l2_relative_error(pred, inputs[:, -1]).item()
                    )
                # Slide window: drop oldest frame, append prediction
                current_window = torch.cat(
                    [current_window[:, 1:], pred.unsqueeze(1)], dim=1
                )

    return {s: float(np.mean(v)) for s, v in step_errors.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="MoE-POT Training")
    parser.add_argument("--mode", choices=["pretrain", "finetune", "downstream"], default="pretrain")
    parser.add_argument("--model_size", choices=["tiny", "small", "medium"], default="tiny")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of datasets")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name for fine-tuning")
    parser.add_argument("--pretrain_ckpt", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    if args.mode == "pretrain":
        cfg = TrainConfig(
            batch_size=args.batch_size,
            lr=args.lr,
        )
        if args.num_epochs is not None:
            cfg.num_epochs = args.num_epochs
        pretrain(
            model_size=args.model_size,
            data_root=args.data_root,
            train_cfg=cfg,
            resume=args.resume,
            local_rank=local_rank,
            world_size=world_size,
        )

    elif args.mode == "finetune":
        assert args.dataset is not None, "--dataset required for fine-tuning"
        assert args.pretrain_ckpt is not None, "--pretrain_ckpt required for fine-tuning"
        cfg = FinetuneConfig(batch_size=args.batch_size, lr=args.lr)
        if args.num_epochs is not None:
            cfg.num_epochs = args.num_epochs
        finetune(
            model_size=args.model_size,
            data_root=args.data_root,
            dataset_name=args.dataset,
            pretrain_ckpt=args.pretrain_ckpt,
            ft_cfg=cfg,
            local_rank=local_rank,
            world_size=world_size,
        )

    elif args.mode == "downstream":
        assert args.dataset is not None, "--dataset required for downstream"
        assert args.pretrain_ckpt is not None, "--pretrain_ckpt required for downstream"
        cfg = DownstreamConfig(batch_size=args.batch_size, lr=args.lr)
        if args.num_epochs is not None:
            cfg.num_epochs = args.num_epochs
        # Downstream uses same fine-tuning logic with longer schedule
        ft_cfg = FinetuneConfig(
            num_epochs=cfg.num_epochs,
            warmup_epochs=cfg.warmup_epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            freeze_router=cfg.freeze_router,
            checkpoint_dir=cfg.checkpoint_dir,
        )
        finetune(
            model_size=args.model_size,
            data_root=args.data_root,
            dataset_name=args.dataset,
            pretrain_ckpt=args.pretrain_ckpt,
            ft_cfg=ft_cfg,
            local_rank=local_rank,
            world_size=world_size,
        )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
