"""
Few-shot finetuning on isotropic Kolmogorov turbulence (Re=222).

Experiment details from the paper (Section 4.4):
  - Dataset: u and v velocity fields at Re=222 [40]
  - 200 training trajectories, 500 test trajectories
  - Finetune FMT-B-42M for 5k steps
  - Joint loss: L(θ, φ, ω) = L_CFM(θ, φ) + λ_VAE * L_VAE(ω)
  - λ_VAE = 1
  - Stop-gradient after VAE encoding (REPA-E style [22])
  - Evaluate with L2RE and VRMSE
"""

import argparse
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from config import FinetuneConfig, get_fmt_config, get_p2vae_config
from data import KolmogorovDataset, collate_traj
from evaluate import compute_metrics, evaluate_rollout, evaluate_reconstruction
from model import FMT, P2VAE, P2VAEWithFMT


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, max_steps: int, base_lr: float, warmup_frac: float = 0.1) -> float:
    warmup_steps = int(max_steps * warmup_frac)
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Load pretrained models
# ---------------------------------------------------------------------------

def load_pretrained(
    vae_ckpt: str,
    fmt_ckpt: str,
    device: torch.device,
) -> Tuple[P2VAE, FMT]:
    vae_state = torch.load(vae_ckpt, map_location=device)
    vae = P2VAE(vae_state["cfg"]).to(device)
    vae.load_state_dict(vae_state["model_state"])

    fmt_state = torch.load(fmt_ckpt, map_location=device)
    fmt = FMT(fmt_state["cfg"]).to(device)
    fmt.load_state_dict(fmt_state["model_state"])

    return vae, fmt


# ---------------------------------------------------------------------------
# Finetuning loop
# ---------------------------------------------------------------------------

def finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = FinetuneConfig(
        data_path=args.data_path,
        n_train_trajs=args.n_train,
        n_test_trajs=args.n_test,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_vae=args.lambda_vae,
        stop_grad_vae=not args.no_stop_grad,
    )

    torch.manual_seed(42)

    # Load pretrained models
    vae, fmt = load_pretrained(args.vae_ckpt, args.fmt_ckpt, device)
    print(f"Loaded P2VAE from {args.vae_ckpt}")
    print(f"Loaded FMT from {args.fmt_ckpt}")

    # Joint model
    joint_model = P2VAEWithFMT(vae, fmt, lambda_vae=cfg.lambda_vae).to(device)

    # Datasets
    train_ds = KolmogorovDataset(
        path=args.data_path,
        traj_len=4,
        split="train",
        n_train=cfg.n_train_trajs,
        n_test=cfg.n_test_trajs,
    )
    test_ds = KolmogorovDataset(
        path=args.data_path,
        traj_len=4,
        split="test",
        n_train=cfg.n_train_trajs,
        n_test=cfg.n_test_trajs,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_traj,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_traj,
    )

    print(f"Train: {len(train_ds)} windows | Test: {len(test_ds)} windows")

    # Optimizer: update all parameters jointly
    optimizer = torch.optim.AdamW(
        joint_model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    scaler = torch.cuda.amp.GradScaler()

    if WANDB_AVAILABLE and args.wandb:
        wandb.init(
            project="fmt-pde",
            name=f"finetune_kolmogorov_{args.n_train}shot",
            config=vars(args),
        )

    os.makedirs("checkpoints", exist_ok=True)

    step = 0
    data_iter = iter(train_loader)

    while step < cfg.max_steps:
        joint_model.train()

        try:
            frames = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            frames = next(data_iter)

        x_seq = [f.to(device) for f in frames]

        current_lr = get_lr(step, cfg.max_steps, cfg.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            loss, info = joint_model(x_seq, stop_grad_vae=cfg.stop_grad_vae)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(joint_model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        step += 1

        if step % 100 == 0:
            print(
                f"Step {step}/{cfg.max_steps} | "
                f"loss={loss.item():.4f} | "
                f"cfm={info.get('loss', torch.tensor(0)).item():.4f} | "
                f"vae={info.get('vae_loss', torch.tensor(0)).item():.4f} | "
                f"lr={current_lr:.2e}"
            )
            if WANDB_AVAILABLE and args.wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/cfm_loss": info.get("loss", torch.tensor(0)).item(),
                    "train/vae_loss": info.get("vae_loss", torch.tensor(0)).item(),
                    "train/lr": current_lr,
                    "step": step,
                })

    # Final evaluation
    print("\n=== Final Evaluation on Kolmogorov Test Set ===")
    vae.eval()
    fmt.eval()

    # Reconstruction quality
    recon_metrics = evaluate_reconstruction(vae, test_loader, device, max_batches=200)
    print(f"Reconstruction — L2RE: {recon_metrics['l2re']:.4f} | VRMSE: {recon_metrics['vrmse']:.4f}")

    # 1-step prediction
    step_errors = evaluate_rollout(
        vae, fmt, test_loader, device,
        n_rollout_steps=1,
        k_val=1.0,
        n_euler_steps=100,
        max_batches=200,
    )
    pred_l2re = float(np.mean(step_errors[0])) if step_errors[0] else float("nan")
    print(f"1-step prediction — L2RE: {pred_l2re:.4f}")

    if WANDB_AVAILABLE and args.wandb:
        wandb.log({
            "test/recon_l2re": recon_metrics["l2re"],
            "test/recon_vrmse": recon_metrics["vrmse"],
            "test/pred_l2re_step1": pred_l2re,
        })

    # Save finetuned checkpoint
    ckpt_path = f"checkpoints/fmt_finetuned_kolmogorov_{args.n_train}shot.pt"
    torch.save({
        "step": step,
        "fmt_state": fmt.state_dict(),
        "vae_state": vae.state_dict(),
        "fmt_cfg": fmt.cfg,
        "vae_cfg": vae.cfg,
    }, ckpt_path)
    print(f"Saved finetuned checkpoint: {ckpt_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-shot finetune on Kolmogorov turbulence")
    parser.add_argument("--vae_ckpt", type=str, required=True,
                        help="Path to pretrained P2VAE checkpoint")
    parser.add_argument("--fmt_ckpt", type=str, required=True,
                        help="Path to pretrained FMT checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to Kolmogorov HDF5 dataset")
    parser.add_argument("--n_train", type=int, default=200,
                        help="Number of training trajectories (few-shot)")
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_vae", type=float, default=1.0)
    parser.add_argument("--no_stop_grad", action="store_true",
                        help="Disable stop-gradient on VAE (not recommended)")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    finetune(args)
