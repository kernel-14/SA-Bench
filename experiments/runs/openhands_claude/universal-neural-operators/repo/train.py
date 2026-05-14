from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import MultiPhysicsDataset, build_dataloader, build_dataset, get_n_in_out
from evaluate import compute_metrics
from model import BaseNeuralOperator, MultiPhysicsModel, build_model, build_multiphysics_model


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred, target)


def nmae_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalized Mean Absolute Error (NMAE) — equation (3) in the paper.
    NMAE = ||G_θ(a) - u||_1 / (max_G(u) - min_G(u) + ε)
    """
    abs_err = (pred - target).abs().mean()
    denom = target.max() - target.min() + eps
    return abs_err / denom


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float = 1.0,
    nmae_weight: float = 0.0,
) -> torch.Tensor:
    loss = mse_weight * mse_loss(pred, target)
    if nmae_weight > 0:
        loss = loss + nmae_weight * nmae_loss(pred, target)
    return loss


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_optimizer(
    params,
    optimizer_type: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> optim.Optimizer:
    if optimizer_type == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def build_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = "cosine",
    n_epochs: int = 100,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
) -> Optional[optim.lr_scheduler._LRScheduler]:
    if scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=min_lr)
    elif scheduler_type == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=n_epochs // 3, gamma=0.5)
    elif scheduler_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


# ---------------------------------------------------------------------------
# Single-physics training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    mse_weight: float = 1.0,
    nmae_weight: float = 0.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_nmae = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = combined_loss(pred, y, mse_weight=mse_weight, nmae_weight=nmae_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse_loss(pred.detach(), y).item()
        total_nmae += nmae_loss(pred.detach(), y).item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "mse": total_mse / n_batches,
        "nmae": total_nmae / n_batches,
    }


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_nmae = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_mse += mse_loss(pred, y).item()
        total_nmae += nmae_loss(pred, y).item()
        n_batches += 1

    return {
        "mse": total_mse / n_batches,
        "nmae": total_nmae / n_batches,
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    device: torch.device,
    save_dir: str,
    model_name: str = "model",
    mse_weight: float = 1.0,
    nmae_weight: float = 0.0,
    log_interval: int = 10,
) -> Dict[str, List[float]]:
    """
    Full training loop for a single-physics model.

    Returns history dict with train/val metrics per epoch.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "train_mse": [], "train_nmae": [], "val_mse": [], "val_nmae": [], "epoch_time": []}
    best_val_mse = float("inf")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, device, mse_weight, nmae_weight)
        val_metrics = eval_epoch(model, val_loader, device)
        epoch_time = time.time() - t0

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["train_mse"].append(train_metrics["mse"])
        history["train_nmae"].append(train_metrics["nmae"])
        history["val_mse"].append(val_metrics["mse"])
        history["val_nmae"].append(val_metrics["nmae"])
        history["epoch_time"].append(epoch_time)

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            torch.save(model.state_dict(), save_path / f"{model_name}_best.pt")

        if epoch % log_interval == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{n_epochs} | "
                f"Train MSE: {train_metrics['mse']:.4e} | "
                f"Val MSE: {val_metrics['mse']:.4e} | "
                f"Val NMAE: {val_metrics['nmae']*100:.4f}% | "
                f"Time: {epoch_time:.2f}s"
            )

    torch.save(model.state_dict(), save_path / f"{model_name}_final.pt")
    return history


# ---------------------------------------------------------------------------
# Multi-physics pre-training loop
# ---------------------------------------------------------------------------

def train_epoch_multiphysics(
    model: MultiPhysicsModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    Training epoch for multi-physics pre-training.
    Batches contain samples from different physics problems.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y, physics_names in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # Handle mixed-physics batches: process each physics separately
        unique_physics = list(set(physics_names))
        batch_loss = torch.tensor(0.0, device=device)

        for phys in unique_physics:
            mask = [i for i, p in enumerate(physics_names) if p == phys]
            x_phys = x[mask]
            y_phys = y[mask]
            pred = model(x_phys, phys)
            batch_loss = batch_loss + mse_loss(pred, y_phys)

        batch_loss = batch_loss / len(unique_physics)
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += batch_loss.item()
        n_batches += 1

    return {"loss": total_loss / n_batches}


def pretrain_multiphysics(
    model: MultiPhysicsModel,
    datasets: Dict[str, Tuple[DataLoader, DataLoader]],
    n_epochs: int,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    device: torch.device,
    save_dir: str,
    model_name: str = "multiphysics",
    log_interval: int = 10,
) -> Dict[str, List[float]]:
    """
    Pre-training on multiple physics problems simultaneously.

    Args:
        datasets: dict mapping physics_name → (train_loader, val_loader)
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Combine all training datasets
    train_datasets = {name: loaders[0].dataset for name, loaders in datasets.items()}
    combined_train = MultiPhysicsDataset(train_datasets)
    combined_loader = DataLoader(
        combined_train,
        batch_size=next(iter(datasets.values()))[0].batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=_multiphysics_collate,
    )

    history = {"train_loss": [], "epoch_time": []}
    best_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        metrics = train_epoch_multiphysics(model, combined_loader, optimizer, device)
        epoch_time = time.time() - t0

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(metrics["loss"])
        history["epoch_time"].append(epoch_time)

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(model.state_dict(), save_path / f"{model_name}_best.pt")

        if epoch % log_interval == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{n_epochs} | "
                f"Train Loss: {metrics['loss']:.4e} | "
                f"Time: {epoch_time:.2f}s"
            )

    torch.save(model.state_dict(), save_path / f"{model_name}_final.pt")
    return history


def _multiphysics_collate(batch):
    """Custom collate for MultiPhysicsDataset that handles variable-size inputs."""
    # Group by physics name
    by_physics: Dict[str, List] = {}
    for x, y, name in batch:
        if name not in by_physics:
            by_physics[name] = []
        by_physics[name].append((x, y))

    # Stack within each physics group
    xs, ys, names = [], [], []
    for name, samples in by_physics.items():
        x_batch = torch.stack([s[0] for s in samples])
        y_batch = torch.stack([s[1] for s in samples])
        xs.append(x_batch)
        ys.append(y_batch)
        names.extend([name] * len(samples))

    # Pad to same size if needed (different physics may have different spatial dims)
    # For simplicity, assume same spatial dims in this implementation
    try:
        x_all = torch.cat(xs, dim=0)
        y_all = torch.cat(ys, dim=0)
    except RuntimeError:
        # If shapes differ, return the first physics group only
        x_all = xs[0]
        y_all = ys[0]
        names = [names[0]] * len(xs[0])

    return x_all, y_all, names


# ---------------------------------------------------------------------------
# Fine-tuning loop
# ---------------------------------------------------------------------------

def finetune(
    model: nn.Module,
    pretrained_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    save_dir: str,
    model_name: str = "finetuned",
    freeze_backbone: bool = True,
    log_interval: int = 10,
) -> Dict[str, List[float]]:
    """
    Fine-tuning with frozen backbone (adapter-only training).

    In the fine-tuning stage, θ_F is fixed; only (θ_{P_ft}, θ_{L_ft}) are trained.
    """
    # Load pre-trained weights
    state_dict = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    if freeze_backbone:
        if hasattr(model, "freeze_backbone"):
            model.freeze_backbone()
        else:
            # Freeze all except lifting and projection
            for name, param in model.named_parameters():
                if "lifting" not in name and "projection" not in name:
                    param.requires_grad = False

    # Only optimize adapter parameters
    if hasattr(model, "adapter_parameters"):
        trainable_params = model.adapter_parameters()
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    print(f"Fine-tuning {sum(p.numel() for p in trainable_params):,} adapter parameters")

    optimizer = build_optimizer(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, n_epochs=n_epochs)

    return train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        model_name=model_name,
        log_interval=log_interval,
    )


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_experiment_from_scratch(cfg: dict, device: torch.device) -> Dict:
    """
    Train a model from scratch on a single physics problem.
    Corresponds to the 'scratch' baseline in Tables 1 and 2.
    """
    dataset_type = cfg["dataset"]["type"]
    t_in = cfg["dataset"].get("t_in", 1)
    n_in, n_out = get_n_in_out(dataset_type, t_in)

    train_ds = build_dataset(dataset_type, cfg["dataset"]["file_path"], split="train",
                             t_in=t_in, **cfg["dataset"].get("kwargs", {}))
    val_ds = build_dataset(dataset_type, cfg["dataset"]["file_path"], split="val",
                           t_in=t_in, **cfg["dataset"].get("kwargs", {}))
    test_ds = build_dataset(dataset_type, cfg["dataset"]["file_path"], split="test",
                            t_in=t_in, **cfg["dataset"].get("kwargs", {}))

    train_loader = build_dataloader(train_ds, batch_size=cfg["training"]["batch_size"])
    val_loader = build_dataloader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)
    test_loader = build_dataloader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    model = build_model(
        cfg["model"]["type"],
        n_in=n_in,
        n_out=n_out,
        spatial_dim=cfg["dataset"].get("spatial_dim", 2),
        **cfg["model"].get("kwargs", {}),
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = build_optimizer(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    scheduler = build_scheduler(optimizer, n_epochs=cfg["training"]["n_epochs"])

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=cfg["training"]["n_epochs"],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=cfg["save_dir"],
        model_name=f"{cfg['model']['type']}_scratch",
    )

    # Final test evaluation
    test_metrics = eval_epoch(model, test_loader, device)
    print(f"Test MSE: {test_metrics['mse']:.4e} | Test NMAE: {test_metrics['nmae']*100:.4f}%")

    return {"history": history, "test_metrics": test_metrics}


def run_experiment_pretrain_finetune(
    pretrain_cfg: dict,
    finetune_cfg: dict,
    device: torch.device,
) -> Dict:
    """
    Pre-train on source physics, then fine-tune on target physics.
    Corresponds to the 'pretr.' rows in Tables 1 and 2.
    """
    # --- Pre-training ---
    src_dataset_type = pretrain_cfg["dataset"]["type"]
    t_in = pretrain_cfg["dataset"].get("t_in", 1)
    n_in_src, n_out_src = get_n_in_out(src_dataset_type, t_in)

    train_ds = build_dataset(src_dataset_type, pretrain_cfg["dataset"]["file_path"],
                             split="train", t_in=t_in)
    val_ds = build_dataset(src_dataset_type, pretrain_cfg["dataset"]["file_path"],
                           split="val", t_in=t_in)

    train_loader = build_dataloader(train_ds, batch_size=pretrain_cfg["training"]["batch_size"])
    val_loader = build_dataloader(val_ds, batch_size=pretrain_cfg["training"]["batch_size"], shuffle=False)

    model = build_model(
        pretrain_cfg["model"]["type"],
        n_in=n_in_src,
        n_out=n_out_src,
        spatial_dim=pretrain_cfg["dataset"].get("spatial_dim", 2),
        **pretrain_cfg["model"].get("kwargs", {}),
    ).to(device)

    print(f"Pre-training {pretrain_cfg['model']['type']} | Parameters: {count_parameters(model):,}")

    optimizer = build_optimizer(
        model.parameters(),
        lr=pretrain_cfg["training"]["lr"],
        weight_decay=pretrain_cfg["training"].get("weight_decay", 1e-4),
    )
    scheduler = build_scheduler(optimizer, n_epochs=pretrain_cfg["training"]["n_epochs"])

    pretrain_history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=pretrain_cfg["training"]["n_epochs"],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=pretrain_cfg["save_dir"],
        model_name=f"{pretrain_cfg['model']['type']}_pretrained",
    )

    pretrained_path = str(Path(pretrain_cfg["save_dir"]) / f"{pretrain_cfg['model']['type']}_pretrained_best.pt")

    # --- Fine-tuning ---
    tgt_dataset_type = finetune_cfg["dataset"]["type"]
    t_in_ft = finetune_cfg["dataset"].get("t_in", 1)
    n_in_tgt, n_out_tgt = get_n_in_out(tgt_dataset_type, t_in_ft)

    ft_train_ds = build_dataset(tgt_dataset_type, finetune_cfg["dataset"]["file_path"],
                                split="train", t_in=t_in_ft)
    ft_val_ds = build_dataset(tgt_dataset_type, finetune_cfg["dataset"]["file_path"],
                              split="val", t_in=t_in_ft)
    ft_test_ds = build_dataset(tgt_dataset_type, finetune_cfg["dataset"]["file_path"],
                               split="test", t_in=t_in_ft)

    ft_train_loader = build_dataloader(ft_train_ds, batch_size=finetune_cfg["training"]["batch_size"])
    ft_val_loader = build_dataloader(ft_val_ds, batch_size=finetune_cfg["training"]["batch_size"], shuffle=False)
    ft_test_loader = build_dataloader(ft_test_ds, batch_size=finetune_cfg["training"]["batch_size"], shuffle=False)

    # Build new model with target physics dimensions
    ft_model = build_model(
        finetune_cfg["model"]["type"],
        n_in=n_in_tgt,
        n_out=n_out_tgt,
        spatial_dim=finetune_cfg["dataset"].get("spatial_dim", 2),
        **finetune_cfg["model"].get("kwargs", {}),
    ).to(device)

    print(f"Fine-tuning {finetune_cfg['model']['type']} | Parameters: {count_parameters(ft_model):,}")

    finetune_history = finetune(
        model=ft_model,
        pretrained_path=pretrained_path,
        train_loader=ft_train_loader,
        val_loader=ft_val_loader,
        n_epochs=finetune_cfg["training"]["n_epochs"],
        lr=finetune_cfg["training"]["lr"],
        weight_decay=finetune_cfg["training"].get("weight_decay", 1e-4),
        device=device,
        save_dir=finetune_cfg["save_dir"],
        model_name=f"{finetune_cfg['model']['type']}_finetuned",
        freeze_backbone=finetune_cfg["training"].get("freeze_backbone", True),
    )

    # Final test evaluation
    test_metrics = eval_epoch(ft_model, ft_test_loader, device)
    print(f"Fine-tune Test MSE: {test_metrics['mse']:.4e} | Test NMAE: {test_metrics['nmae']*100:.4f}%")

    return {
        "pretrain_history": pretrain_history,
        "finetune_history": finetune_history,
        "test_metrics": test_metrics,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Universal Neural Operators - Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--mode", type=str, default="scratch",
                        choices=["scratch", "pretrain", "finetune", "pretrain_finetune"],
                        help="Training mode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    if args.mode == "scratch":
        run_experiment_from_scratch(cfg, device)
    elif args.mode == "pretrain_finetune":
        run_experiment_pretrain_finetune(cfg["pretrain"], cfg["finetune"], device)
    else:
        raise ValueError(f"Mode '{args.mode}' not yet implemented as standalone; use 'pretrain_finetune'")


if __name__ == "__main__":
    main()
