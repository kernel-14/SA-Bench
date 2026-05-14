"""
Training script for WDNO Base-Resolution Model (BRM).

Supports:
  - 1D experiments: Burgers', advection, compressible NS
  - 2D experiments: incompressible fluid, ERA5
  - Both simulation and control modes

Usage:
  python train.py --config configs/burgers_1d.yaml
  python train.py --config configs/burgers_1d.yaml --mode control
  python train.py --config configs/fluid_2d.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet_1d import UNet1D
from models.unet_3d import UNet3D
from models.diffusion import GaussianDiffusion
from utils import (
    set_seed,
    load_config,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    get_cosine_schedule_with_warmup,
    prepare_wavelet_batch_1d,
    prepare_wavelet_batch_2d,
)


def build_model_1d(cfg: Dict[str, Any], mode: str) -> GaussianDiffusion:
    """Build 1D WDNO model (BRM) for 1D PDE experiments."""
    model_cfg = cfg["model"]
    diff_cfg = cfg["diffusion"]
    wt_cfg = cfg["wavelet"]

    n_subbands = wt_cfg["n_subbands_2d"]  # 4 for 2D DWT

    if mode == "simulation":
        # Target: u_traj (1 channel × 4 subbands)
        # Condition: u0 (4 subbands) + f (4 subbands)
        n_state = cfg["data"].get("n_channels", 1)
        in_channels = n_subbands * n_state
        cond_channels = n_subbands * (n_state + 1)  # u0 + f
    else:  # control
        # Target: f (1 channel × 4 subbands)
        # Condition: u0 (4 subbands) + u_T (4 subbands)
        in_channels = n_subbands * 1
        cond_channels = n_subbands * 2

    unet = UNet1D(
        in_channels=in_channels,
        cond_channels=cond_channels,
        init_dim=model_cfg["init_dim"],
        dim_mults=tuple(model_cfg["dim_mults"]),
        resnet_block_groups=model_cfg["resnet_block_groups"],
        attn_heads=model_cfg["attn_heads"],
        attn_dim_head=model_cfg["attn_dim_head"],
    )

    diffusion = GaussianDiffusion(
        model=unet,
        timesteps=diff_cfg["timesteps"],
        schedule=diff_cfg["schedule"],
        ddim_sampling_eta=cfg["inference"]["ddim_eta"],
        p_uncond=diff_cfg["p_uncond"],
    )

    return diffusion


def build_model_2d(cfg: Dict[str, Any], mode: str) -> GaussianDiffusion:
    """Build 2D WDNO model (BRM) for 2D PDE experiments."""
    model_cfg = cfg["model"]
    diff_cfg = cfg["diffusion"]
    wt_cfg = cfg["wavelet"]

    n_subbands = wt_cfg["n_subbands_3d"]  # 8 for 3D DWT
    n_state = cfg["data"].get("n_state_channels", 4)
    n_cond = cfg["data"].get("n_control_channels", 1)

    if mode == "simulation":
        in_channels = n_subbands * n_state
        cond_channels = n_subbands * (1 + n_cond)  # init_density + control
    else:  # control
        in_channels = n_subbands * n_cond
        cond_channels = n_subbands * 1  # init_density

    unet = UNet3D(
        in_channels=in_channels,
        cond_channels=cond_channels,
        init_dim=model_cfg["init_dim"],
        dim_mults=tuple(model_cfg["dim_mults"]),
        resnet_block_groups=model_cfg["resnet_block_groups"],
        attn_heads=model_cfg["attn_heads"],
        attn_dim_head=model_cfg["attn_dim_head"],
    )

    diffusion = GaussianDiffusion(
        model=unet,
        timesteps=diff_cfg["timesteps"],
        schedule=diff_cfg["schedule"],
        ddim_sampling_eta=cfg["inference"]["ddim_eta"],
        p_uncond=diff_cfg["p_uncond"],
    )

    return diffusion


def get_dataset(cfg: Dict[str, Any], split: str, mode: str):
    """Load dataset based on experiment config."""
    exp_name = cfg["experiment"]["name"]

    if "burgers" in exp_name:
        from data.burgers import BurgersDataset
        return BurgersDataset(
            data_path=cfg["data"]["data_path"] if split == "train" else cfg["data"]["test_path"],
            mode=mode,
            split=split,
            n_train=cfg["data"].get("n_train", 9000),
        )
    elif "advection" in exp_name:
        from data.advection import AdvectionDataset
        return AdvectionDataset(
            data_path=cfg["data"]["data_path"] if split == "train" else cfg["data"]["test_path"],
            split=split,
            nx_out=cfg["data"]["nx"],
            nt_out=cfg["data"]["nt"],
        )
    elif "navier_stokes" in exp_name:
        from data.navier_stokes import NavierStokes1DDataset
        return NavierStokes1DDataset(
            data_path=cfg["data"]["data_path"] if split == "train" else cfg["data"]["test_path"],
            split=split,
            nx_out=cfg["data"]["nx"],
            nt_out=cfg["data"]["nt"],
        )
    elif "fluid_2d" in exp_name:
        from data.fluid_2d import Fluid2DDataset
        return Fluid2DDataset(
            data_path=cfg["data"]["data_path"] if split == "train" else cfg["data"]["test_path"],
            mode=mode,
            split=split,
            T=cfg["data"]["T"],
            H=cfg["data"]["H"],
            W=cfg["data"]["W"],
        )
    elif "era5" in exp_name:
        from data.era5 import ERA5Dataset
        return ERA5Dataset(
            data_path=cfg["data"]["data_path"] if split == "train" else cfg["data"]["test_path"],
            split=split,
            n_input=cfg["data"]["n_input"],
            n_output=cfg["data"]["n_output"],
            H=cfg["data"]["H"],
            W=cfg["data"]["W"],
        )
    else:
        raise ValueError(f"Unknown experiment: {exp_name}")


def train_step_1d(
    batch: dict,
    model: GaussianDiffusion,
    cfg: Dict[str, Any],
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    """Single training step for 1D experiments."""
    u = batch["u"].to(device)
    f = batch["f"].to(device)

    wavelet = cfg["wavelet"]["wavelet"]
    wt_mode = cfg["wavelet"]["mode"]

    w_target, w_cond = prepare_wavelet_batch_1d(u, f, wavelet, wt_mode, mode)
    return model(w_target, w_cond)


def train_step_2d(
    batch: dict,
    model: GaussianDiffusion,
    cfg: Dict[str, Any],
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    """Single training step for 2D experiments."""
    state = batch["state"].to(device)       # [B, 4, T, H, W]
    density_init = batch["density_init"].to(device)  # [B, H, W]
    control = batch["control"].to(device)   # [B, T, H, W]

    wavelet = cfg["wavelet"]["wavelet"]
    wt_mode = cfg["wavelet"]["mode"]

    if mode == "simulation":
        # Target: state trajectory
        # Condition: initial density + control
        cond_2d = torch.stack([density_init, control[:, 0]], dim=1)  # [B, 2, H, W]
        # For full control sequence, we need to handle it differently
        # Expand control to [B, T, H, W] and use as condition
        cond_3d = control.unsqueeze(1)  # [B, 1, T, H, W]
        init_3d = density_init.unsqueeze(1).unsqueeze(2).expand(-1, -1, state.shape[2], -1, -1)  # [B, 1, T, H, W]
        cond_full = torch.cat([init_3d, cond_3d], dim=1)  # [B, 2, T, H, W]

        w_target, w_cond = prepare_wavelet_batch_2d(state, density_init.unsqueeze(1), wavelet, wt_mode)
    else:  # control
        # Target: control sequence
        # Condition: initial density
        target_3d = control.unsqueeze(1)  # [B, 1, T, H, W]
        w_target, w_cond = prepare_wavelet_batch_2d(target_3d, density_init.unsqueeze(1), wavelet, wt_mode)

    return model(w_target, w_cond)


def train(cfg: Dict[str, Any], mode: str, resume: Optional[str] = None) -> None:
    """Main training loop."""
    set_seed(cfg["experiment"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = cfg["experiment"]["name"]
    is_2d = cfg["model"]["type"] == "3d"

    # Build model
    if is_2d:
        model = build_model_2d(cfg, mode)
    else:
        model = build_model_1d(cfg, mode)

    model = model.to(device)

    # Multi-GPU support
    n_gpus = cfg["training"].get("n_gpus", 1)
    if n_gpus > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    print(f"Model parameters: {count_parameters(model):,}")

    # Dataset and dataloader
    train_dataset = get_dataset(cfg, "train", mode)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    # LR scheduler (cosine annealing)
    n_steps = cfg["training"]["n_steps"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, n_steps)

    # Resume from checkpoint
    start_step = 0
    if resume is not None:
        ckpt = load_checkpoint(resume, device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step}")

    checkpoint_dir = cfg["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    model.train()
    step = start_step
    data_iter = iter(train_loader)
    best_loss = float("inf")

    pbar = tqdm(range(start_step, n_steps), desc=f"Training {exp_name} ({mode})")
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        optimizer.zero_grad()

        if is_2d:
            loss = train_step_2d(batch, model, cfg, mode, device)
        else:
            loss = train_step_1d(batch, model, cfg, mode, device)

        loss.backward()

        # Gradient clipping
        grad_clip = cfg["training"].get("grad_clip", 1.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        if step % cfg["training"]["log_every"] == 0:
            pbar.set_postfix({"loss": f"{loss.item():.6f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        if step % cfg["training"]["save_every"] == 0 and step > 0:
            state = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": loss.item(),
                "config": cfg,
            }
            save_checkpoint(state, checkpoint_dir, f"step_{step:07d}.pt")

            if loss.item() < best_loss:
                best_loss = loss.item()
                save_checkpoint(state, checkpoint_dir, "best.pt")

    # Save final checkpoint
    state = {
        "step": n_steps,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss": loss.item(),
        "config": cfg,
    }
    save_checkpoint(state, checkpoint_dir, "final.pt")
    print(f"Training complete. Final loss: {loss.item():.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train WDNO Base-Resolution Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--mode", type=str, default=None, help="Override mode: 'simulation' or 'control'")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg["experiment"]["mode"]

    train(cfg, mode, args.resume)


if __name__ == "__main__":
    main()
