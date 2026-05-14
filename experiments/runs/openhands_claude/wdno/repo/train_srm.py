"""
Training script for WDNO Super-Resolution Model (SRM).

The SRM learns p(W_h | W_l, W_a_h) where:
  - W_h: high-resolution wavelet coefficients
  - W_l: low-resolution wavelet coefficients (upsampled to match W_h)
  - W_a_h: high-resolution equation parameter wavelet coefficients

Training data is created by downsampling the original training data:
  - Original: N × M
  - 1x downsampled: (N/2) × (M/2)
  - 2x downsampled: (N/4) × (M/4)
  - etc.

Each batch randomly selects data pairs from a given resolution level.

Usage:
  python train_srm.py --config configs/burgers_1d.yaml
  python train_srm.py --config configs/fluid_2d.yaml
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet_1d import UNet1D
from models.unet_3d import UNet3D
from models.diffusion import GaussianDiffusion
from wavelet.transforms import pad_to_match
from utils import (
    set_seed,
    load_config,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    get_cosine_schedule_with_warmup,
    prepare_wavelet_batch_1d,
)


def build_srm_1d(cfg: Dict[str, Any]) -> GaussianDiffusion:
    """Build SRM for 1D experiments."""
    model_cfg = cfg["model"]
    diff_cfg = cfg["diffusion"]
    n_subbands = cfg["wavelet"]["n_subbands_2d"]

    # SRM target: high-res wavelet coefficients of u_traj
    in_channels = n_subbands * 1  # 4 channels

    # SRM condition: W_l (upsampled, same shape as W_h) + W_a_h (u0 + f at high res)
    # W_l: 4 channels, W_a_h: 8 channels (u0 + f)
    cond_channels = in_channels + n_subbands * 2  # W_l + W_a_h

    unet = UNet1D(
        in_channels=in_channels,
        cond_channels=cond_channels,
        init_dim=model_cfg["init_dim"],
        dim_mults=tuple(model_cfg["dim_mults"]),
        resnet_block_groups=model_cfg["resnet_block_groups"],
        attn_heads=model_cfg["attn_heads"],
        attn_dim_head=model_cfg["attn_dim_head"],
    )

    return GaussianDiffusion(
        model=unet,
        timesteps=diff_cfg["timesteps"],
        schedule=diff_cfg["schedule"],
        ddim_sampling_eta=cfg["inference"]["ddim_eta"],
        p_uncond=diff_cfg["p_uncond"],
    )


def build_srm_2d(cfg: Dict[str, Any]) -> GaussianDiffusion:
    """Build SRM for 2D experiments."""
    model_cfg = cfg["model"]
    diff_cfg = cfg["diffusion"]
    n_subbands = cfg["wavelet"]["n_subbands_3d"]
    n_state = cfg["data"].get("n_state_channels", 4)

    in_channels = n_subbands * n_state  # 32 channels

    # Condition: W_l (same as in_channels) + W_a_h (init_density at high res)
    cond_channels = in_channels + n_subbands * 1

    unet = UNet3D(
        in_channels=in_channels,
        cond_channels=cond_channels,
        init_dim=model_cfg["init_dim"],
        dim_mults=tuple(model_cfg["dim_mults"]),
        resnet_block_groups=model_cfg["resnet_block_groups"],
        attn_heads=model_cfg["attn_heads"],
        attn_dim_head=model_cfg["attn_dim_head"],
    )

    return GaussianDiffusion(
        model=unet,
        timesteps=diff_cfg["timesteps"],
        schedule=diff_cfg["schedule"],
        ddim_sampling_eta=cfg["inference"]["ddim_eta"],
        p_uncond=diff_cfg["p_uncond"],
    )


def train_srm_1d(cfg: Dict[str, Any], resume: Optional[str] = None) -> None:
    """Train SRM for 1D experiments."""
    set_seed(cfg["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from data.burgers import BurgersMultiResDataset

    dataset = BurgersMultiResDataset(
        data_path=cfg["data"]["data_path"],
        n_train=cfg["data"].get("n_train", 9000),
        max_levels=cfg.get("super_resolution", {}).get("max_sr_levels", 3),
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    model = build_srm_1d(cfg).to(device)
    print(f"SRM parameters: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    n_steps = cfg["training"]["n_steps"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, n_steps)

    start_step = 0
    if resume is not None:
        ckpt = load_checkpoint(resume, device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]

    checkpoint_dir = cfg["training"]["checkpoint_dir"] + "_srm"
    os.makedirs(checkpoint_dir, exist_ok=True)

    wavelet = cfg["wavelet"]["wavelet"]
    wt_mode = cfg["wavelet"]["mode"]

    from pytorch_wavelets import DWTForward, DWT1DForward

    def get_wavelet_coeffs(u_high, f_high, u_low, f_low):
        """Get wavelet coefficients for high and low resolution data."""
        device_ = u_high.device
        dwt2d = DWTForward(J=1, wave=wavelet, mode=wt_mode).to(device_)
        dwt1d = DWT1DForward(J=1, wave=wavelet, mode=wt_mode).to(device_)

        def wt2d(x):
            x_c = x.unsqueeze(1)
            yl, yh = dwt2d(x_c)
            return torch.cat([yl, yh[0][:, :, 0], yh[0][:, :, 1], yh[0][:, :, 2]], dim=1)

        def wt1d_to_2d(x, T_prime, X_prime):
            x_c = x.unsqueeze(1)
            yl, yh = dwt1d(x_c)
            cD = yh[0][:, :, 0]
            coeffs = torch.cat([yl, cD, yl, cD], dim=1)
            return coeffs.unsqueeze(2).expand(-1, -1, T_prime, -1)

        # High-res wavelet coefficients
        w_u_high = wt2d(u_high[:, 1:])  # [B, 4, T', X']
        T_prime, X_prime = w_u_high.shape[-2], w_u_high.shape[-1]
        w_u0_high = wt1d_to_2d(u_high[:, 0], T_prime, X_prime)
        w_f_high = wt2d(f_high)
        w_a_high = torch.cat([w_u0_high, w_f_high], dim=1)  # [B, 8, T', X']

        # Low-res wavelet coefficients
        w_u_low = wt2d(u_low[:, 1:])  # [B, 4, T'', X'']

        # Upsample low-res to match high-res
        w_u_low_up = pad_to_match(w_u_low, w_u_high)  # [B, 4, T', X']

        return w_u_high, w_u_low_up, w_a_high

    model.train()
    data_iter = iter(loader)
    best_loss = float("inf")

    pbar = tqdm(range(start_step, n_steps), desc="Training SRM (1D)")
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        u_high = batch["u_high"].to(device)
        u_low = batch["u_low"].to(device)
        f_high = batch["f_high"].to(device)
        f_low = batch["f_low"].to(device)

        w_u_high, w_u_low_up, w_a_high = get_wavelet_coeffs(u_high, f_high, u_low, f_low)

        # SRM condition: W_l (upsampled) + W_a_h
        cond = torch.cat([w_u_low_up, w_a_high], dim=1)

        optimizer.zero_grad()
        loss = model(w_u_high, cond)
        loss.backward()

        grad_clip = cfg["training"].get("grad_clip", 1.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        if step % cfg["training"]["log_every"] == 0:
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        if step % cfg["training"]["save_every"] == 0 and step > 0:
            state = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": loss.item(),
            }
            save_checkpoint(state, checkpoint_dir, f"step_{step:07d}.pt")
            if loss.item() < best_loss:
                best_loss = loss.item()
                save_checkpoint(state, checkpoint_dir, "best.pt")

    state = {
        "step": n_steps,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss": loss.item(),
    }
    save_checkpoint(state, checkpoint_dir, "final.pt")
    print(f"SRM training complete. Final loss: {loss.item():.6f}")


def train_srm_2d(cfg: Dict[str, Any], resume: Optional[str] = None) -> None:
    """Train SRM for 2D experiments."""
    set_seed(cfg["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from data.fluid_2d import Fluid2DMultiResDataset

    dataset = Fluid2DMultiResDataset(
        data_path=cfg["data"]["data_path"],
        max_levels=cfg.get("super_resolution", {}).get("max_sr_levels", 2),
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    model = build_srm_2d(cfg).to(device)
    print(f"SRM parameters: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    n_steps = cfg["training"]["n_steps"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, n_steps)

    start_step = 0
    if resume is not None:
        ckpt = load_checkpoint(resume, device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]

    checkpoint_dir = cfg["training"]["checkpoint_dir"] + "_srm"
    os.makedirs(checkpoint_dir, exist_ok=True)

    wavelet = cfg["wavelet"]["wavelet"]
    wt_mode = cfg["wavelet"]["mode"]

    import ptwt
    import pywt as pywt_lib

    def wt3d(x):
        """x: [B, C, T, H, W] → [B, 8*C, T', H', W']"""
        w = pywt_lib.Wavelet(wavelet)
        coeffs = ptwt.wavedec3(x, w, level=1, mode=wt_mode)
        approx = coeffs[0]
        d = coeffs[1]
        return torch.cat([approx, d['aad'], d['ada'], d['add'], d['daa'], d['dad'], d['dda'], d['ddd']], dim=1)

    model.train()
    data_iter = iter(loader)
    best_loss = float("inf")

    pbar = tqdm(range(start_step, n_steps), desc="Training SRM (2D)")
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        state_high = batch["state_high"].to(device)  # [B, 4, T, H, W]
        state_low = batch["state_low"].to(device)    # [B, 4, T, H', W'] (H'=H/2)

        # Wavelet transform
        w_high = wt3d(state_high)  # [B, 32, T', H'', W'']
        w_low = wt3d(state_low)    # [B, 32, T', H''', W''']

        # Upsample low-res to match high-res
        w_low_up = pad_to_match(w_low, w_high)

        # Condition: W_l (upsampled) + W_a_h (init density at high res)
        # Use first channel of state as "equation parameter" (initial density)
        init_density_high = state_high[:, 0:1, 0:1, :, :]  # [B, 1, 1, H, W]
        init_density_high = init_density_high.expand(-1, -1, state_high.shape[2], -1, -1)  # [B, 1, T, H, W]
        w_a_high = wt3d(init_density_high)  # [B, 8, T', H'', W'']

        cond = torch.cat([w_low_up, w_a_high], dim=1)

        optimizer.zero_grad()
        loss = model(w_high, cond)
        loss.backward()

        grad_clip = cfg["training"].get("grad_clip", 1.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        if step % cfg["training"]["log_every"] == 0:
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        if step % cfg["training"]["save_every"] == 0 and step > 0:
            state = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": loss.item(),
            }
            save_checkpoint(state, checkpoint_dir, f"step_{step:07d}.pt")
            if loss.item() < best_loss:
                best_loss = loss.item()
                save_checkpoint(state, checkpoint_dir, "best.pt")

    state = {
        "step": n_steps,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss": loss.item(),
    }
    save_checkpoint(state, checkpoint_dir, "final.pt")
    print(f"SRM training complete. Final loss: {loss.item():.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train WDNO Super-Resolution Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    is_2d = cfg["model"]["type"] == "3d"

    if is_2d:
        train_srm_2d(cfg, args.resume)
    else:
        train_srm_1d(cfg, args.resume)


if __name__ == "__main__":
    main()
