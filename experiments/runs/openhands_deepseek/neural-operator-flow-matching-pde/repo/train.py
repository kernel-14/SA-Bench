"""Training and evaluation for the generative PDE foundation model.

Two-stage training:
  1. Train P2VAE (100k steps) for latent compression
  2. Train FMT (100k steps) for flow marching with frozen P2VAE

Also supports fine-tuning on downstream tasks and evaluation.
"""

import argparse
import math
import os
import random
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from config import (
    Config, get_p2vae_config, get_fmt_config,
    P2VAEConfig, FMTConfig, TrainingConfig, DataConfig, FlowMarchingConfig,
)
from modules import (
    P2VAE, FMT,
    sample_x_t_k, flow_marching_loss_fn, euler_sampler_fmt,
    autoregressive_predict, autoregressive_rollout, generate_ensemble,
    spatial_downsample, rearrange_velocities_to_frames,
)
from data import (
    build_dataloaders, MultiPDEDataset, FlowMarchingDataset,
    compute_l2re, compute_vrmse,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
):
    """Cosine learning rate schedule with linear warmup."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Stage 1: P2VAE Training
# ---------------------------------------------------------------------------

def train_p2vae(
    config: Config,
    model_size: str = "16M",
    resume_from: Optional[str] = None,
):
    """Train P2VAE on PDE frames."""
    p2vae_cfg = get_p2vae_config(model_size)
    train_cfg = config.training
    data_cfg = config.data

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    model = P2VAE(
        in_channels=data_cfg.in_channels,
        base_dim=p2vae_cfg.base_dim,
        channel_mult=p2vae_cfg.channel_mult,
        num_res_blocks=p2vae_cfg.num_res_blocks,
        attention_resolutions=p2vae_cfg.attention_resolutions,
        z_channels=p2vae_cfg.z_channels,
        dropout=p2vae_cfg.dropout,
        kl_weight=train_cfg.p2vae_kl_weight,
    ).to(device)

    if resume_from:
        model.load_state_dict(torch.load(resume_from, map_location=device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.p2vae_lr,
        betas=(train_cfg.p2vae_beta1, train_cfg.p2vae_beta2),
        weight_decay=train_cfg.p2vae_weight_decay,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        train_cfg.p2vae_warmup_steps,
        train_cfg.p2vae_steps,
    )

    scaler = GradScaler() if config.use_amp else None

    # P2VAE trains on individual frames, not sequences
    class FrameDataset(torch.utils.data.Dataset):
        def __init__(self, source_dataset, num_frames=5):
            self.source = source_dataset
            self.num_frames = num_frames

        def __len__(self):
            return len(self.source) * self.num_frames

        def __getitem__(self, idx):
            traj_idx = idx // self.num_frames
            frame_idx = idx % self.num_frames
            traj = self.source[traj_idx]
            return traj[frame_idx]

    base_dataloader = build_dataloaders(
        data_cfg, split="train",
        batch_size=train_cfg.p2vae_batch_size,
        num_workers=train_cfg.num_workers,
    )
    frame_ds = FrameDataset(base_dataloader.dataset.source_dataset.source_dataset)
    dataloader = DataLoader(
        frame_ds,
        batch_size=train_cfg.p2vae_batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_dataloader = build_dataloaders(
        data_cfg, split="val",
        batch_size=train_cfg.p2vae_batch_size,
        num_workers=train_cfg.num_workers,
    )

    model.train()
    global_step = 0
    best_val_loss = float("inf")
    checkpoint_dir = Path("./checkpoints/p2vae")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training P2VAE-{model_size} for {train_cfg.p2vae_steps} steps")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    while global_step < train_cfg.p2vae_steps:
        for batch in dataloader:
            if global_step >= train_cfg.p2vae_steps:
                break

            x = batch.to(device)

            with autocast() if config.use_amp else torch.enable_grad():
                recon, z, recon_loss, kl_loss = model(x)
                loss = recon_loss + kl_loss

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                if train_cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if train_cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optimizer.step()

            scheduler.step()
            global_step += 1

            if global_step % 100 == 0:
                print(
                    f"Step {global_step:6d} | Loss: {loss.item():.4f} "
                    f"| Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f} "
                    f"| LR: {scheduler.get_last_lr()[0]:.2e}"
                )

            if global_step % 5000 == 0:
                model.eval()
                val_loss_total = 0.0
                val_count = 0
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        x_val = val_batch["x0"].to(device) if isinstance(val_batch, dict) else val_batch.to(device)
                        recon_v, _, recon_l, kl_l = model(x_val)
                        val_loss_total += (recon_l + kl_l).item()
                        val_count += 1
                        if val_count >= 20:
                            break
                avg_val_loss = val_loss_total / val_count
                print(f"  Val Loss: {avg_val_loss:.4f}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(
                        model.state_dict(),
                        checkpoint_dir / f"p2vae_{model_size}_best.pt",
                    )
                model.train()

    # Save final checkpoint
    torch.save(
        model.state_dict(),
        checkpoint_dir / f"p2vae_{model_size}_final.pt",
    )
    print(f"P2VAE training complete. Saved to {checkpoint_dir}")

    return model


# ---------------------------------------------------------------------------
# Stage 2: FMT Training
# ---------------------------------------------------------------------------

def train_fmt(
    config: Config,
    p2vae_ckpt: str,
    fmt_size: str = "B",
    resume_from: Optional[str] = None,
):
    """Train FMT with frozen P2VAE encoder.

    Training objective (Eq. 12):
      L_CFM = 0.5 * E_{h_s ~ p_phi(h_s | h_{s-1}, x_{s,t_s}^{k_s}, t_s)}
        sum_{s=1}^{T} ||(1-t_s) g(x_{s,t_s}^{k_s}, t_s, h_{s-1}) - (x_{s+1} - x_{s,t_s}^{k_s})||^2
    """
    p2vae_cfg = get_p2vae_config("16M")
    fmt_cfg = get_fmt_config(fmt_size)
    train_cfg = config.training
    data_cfg = config.data

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Load frozen P2VAE
    p2vae = P2VAE(
        in_channels=data_cfg.in_channels,
        base_dim=p2vae_cfg.base_dim,
        channel_mult=p2vae_cfg.channel_mult,
        num_res_blocks=p2vae_cfg.num_res_blocks,
        attention_resolutions=p2vae_cfg.attention_resolutions,
        z_channels=p2vae_cfg.z_channels,
        dropout=p2vae_cfg.dropout,
    ).to(device)
    p2vae.load_state_dict(torch.load(p2vae_ckpt, map_location=device))
    p2vae.eval()
    for param in p2vae.parameters():
        param.requires_grad = False

    model = FMT(
        latent_channels=fmt_cfg.latent_channels,
        latent_spatial_size=fmt_cfg.latent_spatial_size,
        embed_dim=fmt_cfg.embed_dim,
        num_heads=fmt_cfg.num_heads,
        depth=fmt_cfg.depth,
        mlp_ratio=fmt_cfg.mlp_ratio,
        pyramid_ratios=fmt_cfg.pyramid_ratios,
        rnn_hidden_dim=fmt_cfg.rnn_hidden_dim,
    ).to(device)

    if resume_from:
        model.load_state_dict(torch.load(resume_from, map_location=device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.fmt_lr,
        betas=(train_cfg.fmt_beta1, train_cfg.fmt_beta2),
        weight_decay=train_cfg.fmt_weight_decay,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        train_cfg.fmt_warmup_steps,
        train_cfg.fmt_steps,
    )

    scaler = GradScaler() if config.use_amp else None

    dataloader = build_dataloaders(
        data_cfg, split="train",
        batch_size=train_cfg.fmt_batch_size,
        num_workers=train_cfg.num_workers,
    )

    val_dataloader = build_dataloaders(
        data_cfg, split="val",
        batch_size=train_cfg.fmt_batch_size,
        num_workers=train_cfg.num_workers,
    )

    checkpoint_dir = Path("./checkpoints/fmt")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    global_step = 0
    best_val_loss = float("inf")

    print(f"Training FMT-{fmt_size} for {train_cfg.fmt_steps} steps")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    num_frames = config.data.trajectory_length  # Typically 5 (x0..x4), 4 transitions

    while global_step < train_cfg.fmt_steps:
        for batch in dataloader:
            if global_step >= train_cfg.fmt_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            B = batch["x0"].shape[0]

            # Encode frames to latent space
            y_frames = []
            with torch.no_grad():
                for s in range(num_frames):
                    mu_s, logvar_s = p2vae.encode(batch[f"x{s}"])
                    y_s = p2vae.reparameterize(mu_s, logvar_s)
                    y_frames.append(y_s)

            # Build noisy latents for frames 0..3 (transitions to frames 1..4)
            y_noisy_list = []
            t_list = []
            for s in range(num_frames - 1):
                y_s = y_frames[s]
                y_next = y_frames[s + 1]
                t_s = batch[f"t{s}"].view(B, 1, 1, 1)
                k_s = batch[f"k{s}"].view(B, 1, 1, 1)
                z = torch.randn_like(y_s)
                y_noisy = t_s * y_next + k_s * (1 - t_s) * y_s + (1 - t_s) * (1 - k_s) * z
                y_noisy_list.append(y_noisy)
                t_list.append(batch[f"t{s}"].view(B))

            y_noisy_tensor = torch.stack(y_noisy_list, dim=1)  # [B, 4, C, H, W]
            t_tensor = torch.stack(t_list, dim=1)  # [B, 4]
            h_init = torch.zeros(B, model.gru.hidden_dim, device=device)

            with autocast() if config.use_amp else torch.enable_grad():
                velocities, _ = model(y_noisy_tensor, t_tensor, h_init)

                losses = []
                offset = 0
                for s in range(num_frames - 1):
                    ratio = model.pyramid_ratios[s]
                    n_tokens = (model.latent_spatial_size // ratio) ** 2

                    vel_s = velocities[:, offset:offset + n_tokens]

                    target_y_next = spatial_downsample(y_frames[s + 1], ratio)
                    target_y_curr = spatial_downsample(y_noisy_tensor[:, s], ratio)

                    Bp, Cp, Hp, Wp = target_y_curr.shape
                    tgt_curr_flat = target_y_curr.permute(0, 2, 3, 1).reshape(Bp, Hp * Wp, Cp)
                    tgt_next_flat = target_y_next.permute(0, 2, 3, 1).reshape(Bp, Hp * Wp, Cp)

                    t_s_val = batch[f"t{s}"].view(B, 1, 1)
                    frame_loss = flow_marching_loss_fn(
                        vel_s, tgt_curr_flat, tgt_next_flat, t_s_val,
                    )
                    losses.append(frame_loss)
                    offset += n_tokens

                loss = torch.stack(losses).mean()

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                if train_cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if train_cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optimizer.step()

            scheduler.step()
            global_step += 1

            if global_step % 100 == 0:
                print(
                    f"Step {global_step:6d} | Loss: {loss.item():.4f} "
                    f"| LR: {scheduler.get_last_lr()[0]:.2e}"
                )

            if global_step % 5000 == 0:
                model.eval()
                val_loss_total = 0.0
                val_count = 0
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        val_batch = {k: v.to(device) for k, v in val_batch.items()}
                        Bv = val_batch["x0"].shape[0]

                        y_val = []
                        for s in range(num_frames):
                            mu_s, _ = p2vae.encode(val_batch[f"x{s}"])
                            y_val.append(mu_s)

                        y_noisy_v = []
                        t_val_list = []
                        for s in range(num_frames - 1):
                            t_s = val_batch[f"t{s}"].view(Bv, 1, 1, 1)
                            k_s = val_batch[f"k{s}"].view(Bv, 1, 1, 1)
                            z = torch.randn_like(y_val[s])
                            y_noisy_v.append(
                                t_s * y_val[s + 1] + k_s * (1 - t_s) * y_val[s] + (1 - t_s) * (1 - k_s) * z
                            )
                            t_val_list.append(val_batch[f"t{s}"].view(Bv))

                        y_noisy_vt = torch.stack(y_noisy_v, dim=1)
                        t_vt = torch.stack(t_val_list, dim=1)
                        h_init_v = torch.zeros(Bv, model.gru.hidden_dim, device=device)

                        velocities_v, _ = model(y_noisy_vt, t_vt, h_init_v)

                        v_losses = []
                        offset_v = 0
                        for s in range(num_frames - 1):
                            ratio = model.pyramid_ratios[s]
                            n_tokens = (model.latent_spatial_size // ratio) ** 2
                            vel_v = velocities_v[:, offset_v:offset_v + n_tokens]
                            tgt_next_d = spatial_downsample(y_val[s + 1], ratio)
                            tgt_curr_d = spatial_downsample(y_noisy_vt[:, s], ratio)
                            Bp, Cp, Hp, Wp = tgt_curr_d.shape
                            v_losses.append(flow_marching_loss_fn(
                                vel_v,
                                tgt_curr_d.permute(0, 2, 3, 1).reshape(Bp, Hp * Wp, Cp),
                                tgt_next_d.permute(0, 2, 3, 1).reshape(Bp, Hp * Wp, Cp),
                                val_batch[f"t{s}"].view(Bv, 1, 1),
                            ))
                            offset_v += n_tokens

                        val_loss_total += torch.stack(v_losses).mean().item()
                        val_count += 1
                        if val_count >= 20:
                            break

                avg_val_loss = val_loss_total / max(val_count, 1)
                print(f"  Val Loss: {avg_val_loss:.4f}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(
                        model.state_dict(),
                        checkpoint_dir / f"fmt_{fmt_size}_best.pt",
                    )
                model.train()

    torch.save(
        model.state_dict(),
        checkpoint_dir / f"fmt_{fmt_size}_final.pt",
    )
    print(f"FMT training complete. Saved to {checkpoint_dir}")

    return model


# ---------------------------------------------------------------------------
# Fine-tuning on downstream task
# ---------------------------------------------------------------------------

def finetune_kolmogorov(
    config: Config,
    p2vae_ckpt: str,
    fmt_ckpt: str,
    fmt_size: str = "B",
    train_data_path: str = "./data/kolmogorov_train.h5",
    val_data_path: str = "./data/kolmogorov_test.h5",
):
    """Fine-tune on Kolmogorov turbulence with stop-gradient (REPA-E style).

    Loss = L_CFM + λ_VAE * L_VAE
    """
    p2vae_cfg = get_p2vae_config("16M")
    fmt_cfg = get_fmt_config(fmt_size)
    train_cfg = config.training

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    p2vae = P2VAE(
        in_channels=config.data.in_channels,
        base_dim=p2vae_cfg.base_dim,
        channel_mult=p2vae_cfg.channel_mult,
        num_res_blocks=p2vae_cfg.num_res_blocks,
        attention_resolutions=p2vae_cfg.attention_resolutions,
        z_channels=p2vae_cfg.z_channels,
    ).to(device)
    p2vae.load_state_dict(torch.load(p2vae_ckpt, map_location=device))

    model = FMT(
        latent_channels=fmt_cfg.latent_channels,
        latent_spatial_size=fmt_cfg.latent_spatial_size,
        embed_dim=fmt_cfg.embed_dim,
        num_heads=fmt_cfg.num_heads,
        depth=fmt_cfg.depth,
        mlp_ratio=fmt_cfg.mlp_ratio,
        pyramid_ratios=fmt_cfg.pyramid_ratios,
        rnn_hidden_dim=fmt_cfg.rnn_hidden_dim,
    ).to(device)
    model.load_state_dict(torch.load(fmt_ckpt, map_location=device))

    params = list(model.parameters()) + list(p2vae.decoder.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=train_cfg.finetune_lr, betas=(0.9, 0.95), weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, train_cfg.finetune_steps // 10, train_cfg.finetune_steps,
    )
    scaler = GradScaler() if config.use_amp else None

    import h5py
    with h5py.File(train_data_path, "r") as f:
        train_data = torch.from_numpy(f["trajectories"][:200]).float()
    with h5py.File(val_data_path, "r") as f:
        val_data = torch.from_numpy(f["trajectories"][:500]).float()

    model.train()
    p2vae.train()
    best_val_l2re = float("inf")
    checkpoint_dir = Path("./checkpoints/kolmogorov")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fine-tuning on Kolmogorov turbulence for {train_cfg.finetune_steps} steps")

    for step in range(train_cfg.finetune_steps):
        idx = random.randint(0, len(train_data) - 1)
        start = random.randint(0, train_data.shape[1] - 5)
        traj = train_data[idx, start:start + 5].to(device)  # [5, C, H, W]
        B = 1

        with torch.no_grad():
            mu_list = [p2vae.encode(traj[s:s+1])[0].detach() for s in range(5)]
        y_frames = mu_list

        y_noisy_list = []
        t_list = []
        for s in range(4):
            t_v = random.random()
            k_v = random.random()
            t_list.append(t_v)
            z = torch.randn_like(y_frames[s])
            y_noisy = t_v * y_frames[s + 1] + k_v * (1 - t_v) * y_frames[s] + (1 - t_v) * (1 - k_v) * z
            y_noisy_list.append(y_noisy)

        y_noisy_t = torch.stack(y_noisy_list, dim=1)
        t_t = torch.tensor(t_list, device=device).unsqueeze(0)
        h_init = torch.zeros(B, model.gru.hidden_dim, device=device)

        with autocast() if config.use_amp else torch.enable_grad():
            velocities, _ = model(y_noisy_t, t_t, h_init)

            cfm_losses = []
            offset = 0
            for s in range(4):
                ratio = model.pyramid_ratios[s]
                n_tokens = (model.latent_spatial_size // ratio) ** 2
                vel_s = velocities[:, offset:offset + n_tokens]
                tgt_next = spatial_downsample(y_frames[s + 1], ratio)
                tgt_curr = spatial_downsample(y_noisy_t[:, s], ratio)
                Bp, Cp, Hp, Wp = tgt_curr.shape
                t_s_v = torch.tensor(t_list[s], device=device).view(1, 1, 1)
                cfm_losses.append(flow_marching_loss_fn(
                    vel_s,
                    tgt_curr.permute(0, 2, 3, 1).reshape(Bp, Hp * Wp, Cp),
                    tgt_next.permute(0, 2, 3, 1).reshape(Bp, Hp * Wp, Cp),
                    t_s_v,
                ))
                offset += n_tokens
            cfm_loss = torch.stack(cfm_losses).mean()

            recon, _, recon_loss, kl_loss = p2vae(traj[0:1])
            total_loss = cfm_loss + train_cfg.lambda_vae * (recon_loss + kl_loss)

        optimizer.zero_grad()
        if scaler:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
        scheduler.step()

        if step % 50 == 0:
            print(f"Step {step:5d} | Total: {total_loss.item():.4f} | CFM: {cfm_loss.item():.4f}")

        if step % 500 == 0:
            model.eval()
            p2vae.eval()
            l2re_total = 0.0
            with torch.no_grad():
                for ei in range(min(10, len(val_data))):
                    et = val_data[ei:ei+1, 0:5].to(device)
                    x_pred = autoregressive_predict(
                        p2vae, model, et[0, :4],
                        config.flow_marching.num_sampling_steps,
                        config.flow_marching.dt, device,
                    )
                    l2re_total += compute_l2re(x_pred, et[0, 4:5]).item()
            avg_l2re = l2re_total / min(10, len(val_data))
            print(f"  Val L2RE: {avg_l2re:.4f}")
            if avg_l2re < best_val_l2re:
                best_val_l2re = avg_l2re
                torch.save(model.state_dict(), checkpoint_dir / "fmt_kolmogorov_best.pt")
                torch.save(p2vae.state_dict(), checkpoint_dir / "p2vae_kolmogorov_best.pt")
            model.train()
            p2vae.train()

    torch.save(model.state_dict(), checkpoint_dir / "fmt_kolmogorov_final.pt")
    torch.save(p2vae.state_dict(), checkpoint_dir / "p2vae_kolmogorov_final.pt")
    return model, p2vae


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    config: Config,
    p2vae_ckpt: str,
    fmt_ckpt: str,
    fmt_size: str = "B",
    dataset_name: str = "PA-NS",
    data_path: str = "./data/PA-NS.h5",
):
    """Evaluate model on a specific PDE dataset for long-term rollout."""
    p2vae_cfg = get_p2vae_config("16M")
    fmt_cfg = get_fmt_config(fmt_size)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    p2vae = P2VAE(
        in_channels=config.data.in_channels,
        base_dim=p2vae_cfg.base_dim,
        channel_mult=p2vae_cfg.channel_mult,
        num_res_blocks=p2vae_cfg.num_res_blocks,
        attention_resolutions=p2vae_cfg.attention_resolutions,
        z_channels=p2vae_cfg.z_channels,
    ).to(device)
    p2vae.load_state_dict(torch.load(p2vae_ckpt, map_location=device))
    p2vae.eval()

    fmt = FMT(
        latent_channels=fmt_cfg.latent_channels,
        latent_spatial_size=fmt_cfg.latent_spatial_size,
        embed_dim=fmt_cfg.embed_dim,
        num_heads=fmt_cfg.num_heads,
        depth=fmt_cfg.depth,
        mlp_ratio=fmt_cfg.mlp_ratio,
        pyramid_ratios=fmt_cfg.pyramid_ratios,
        rnn_hidden_dim=fmt_cfg.rnn_hidden_dim,
    ).to(device)
    fmt.load_state_dict(torch.load(fmt_ckpt, map_location=device))
    fmt.eval()

    import h5py
    with h5py.File(data_path, "r") as f:
        data = torch.from_numpy(f["trajectories"][:]).float()

    eval_steps = [1, 5, 10, -1]  # -1 means last step
    results = {}

    for eval_step in eval_steps:
        l2re_values = []
        max_trajs = min(50, len(data))
        for traj_idx in range(max_trajs):
            traj = data[traj_idx].to(device)
            T_avail = traj.shape[0]
            if eval_step == -1:
                rollout_len = min(T_avail - 4, 20)
            else:
                rollout_len = eval_step

            if rollout_len <= 0 or T_avail < 4 + rollout_len:
                continue

            x_init = traj[:4]
            preds = autoregressive_rollout(
                p2vae, fmt, x_init, rollout_len,
                config.flow_marching.num_sampling_steps,
                config.flow_marching.dt,
                device,
            )

            idx = rollout_len - 1 if eval_step == -1 else eval_step - 1
            pred_at_step = preds[idx:idx+1]
            target_at_step = traj[4 + idx:4 + idx + 1]
            l2re_values.append(compute_l2re(
                pred_at_step.to(device), target_at_step.to(device)
            ).item())

        step_name = "last" if eval_step == -1 else f"step_{eval_step}"
        avg_l2re = np.mean(l2re_values) if l2re_values else 0.0
        results[step_name] = avg_l2re
        print(f"  {dataset_name} {step_name}: L2RE = {avg_l2re:.4f}")

    avg_all = np.mean(list(results.values())) if results else 0.0
    results["average"] = avg_all
    print(f"  {dataset_name} average: L2RE = {avg_all:.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generative PDE Foundation Model")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="Training stage: 1=P2VAE, 2=FMT")
    parser.add_argument("--model_size", type=str, default="16M",
                        help="Model size (P2VAE: 16M/87M, FMT: S/B/L)")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--finetune", action="store_true", help="Run Kolmogorov fine-tuning")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for resume/evaluation")
    parser.add_argument("--p2vae_checkpoint", type=str, default=None,
                        help="P2VAE checkpoint for FMT training")
    parser.add_argument("--dataset", type=str, default="PA-NS",
                        help="Dataset name for evaluation")
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to dataset directory")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = Config()
    config.seed = args.seed
    set_seed(args.seed)

    if args.eval:
        p2vae_ckpt = args.p2vae_checkpoint or f"./checkpoints/p2vae/p2vae_16M_best.pt"
        fmt_ckpt = args.checkpoint or f"./checkpoints/fmt/fmt_B_best.pt"
        evaluate_model(config, p2vae_ckpt, fmt_ckpt, args.model_size,
                       args.dataset, args.data_path)
    elif args.finetune:
        p2vae_ckpt = args.p2vae_checkpoint or f"./checkpoints/p2vae/p2vae_16M_best.pt"
        fmt_ckpt = args.checkpoint or f"./checkpoints/fmt/fmt_B_best.pt"
        finetune_kolmogorov(config, p2vae_ckpt, fmt_ckpt, args.model_size)
    elif args.stage == 1:
        train_p2vae(config, args.model_size, args.checkpoint)
    elif args.stage == 2:
        p2vae_ckpt = args.p2vae_checkpoint or f"./checkpoints/p2vae/p2vae_16M_best.pt"
        train_fmt(config, p2vae_ckpt, args.model_size, args.checkpoint)


if __name__ == "__main__":
    main()
