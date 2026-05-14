"""
Few-shot finetuning script for Kolmogorov turbulence adaptation.

From the paper:
- Finetune pretrained P2VAE + FMT-B-42M on 200 training trajectories
- 5k steps
- End-to-end finetuning with stop-gradient after latent generation
- Loss: L(theta, phi, omega) = L_CFM(theta, phi) + lambda_VAE * L_VAE(omega)
- lambda_VAE = 1
- Test on 500 trajectories

The stop-gradient prevents the CFM loss from deteriorating the autoencoder.
Following REPA-E (Leng et al., 2025).

Dataset: Kolmogorov flow at Re=222 with u and v fields
(Sardar & Skillen, 2025)
"""

import os
import math
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.p2vae import P2VAE, P2VAE_16M
from models.fmt import FlowMarchingTransformer, FMTBase
from models.flow_marching import sample_interpolation, flow_marching_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
):
    """Cosine learning rate schedule with linear warmup."""

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class KolmogorovDataset(Dataset):
    """
    Dataset for Kolmogorov turbulence at Re=222.

    Loads u and v velocity fields and returns 4-frame trajectories.
    Fields are resized to 128x128 and padded to 3 channels (u, v, 0).
    """

    def __init__(self, data_path: str, split: str = "train", num_trajectories: int = None):
        """
        Args:
            data_path: path to the Kolmogorov dataset
            split: 'train' or 'test'
            num_trajectories: limit number of trajectories (200 for train, 500 for test)
        """
        self.data_path = Path(data_path)
        self.split = split
        self.num_trajectories = num_trajectories

        # Load data - expected format: (N_traj, T, H, W) for each field
        # or (N_traj, T, C, H, W)
        self._load_data()

    def _load_data(self):
        """Load Kolmogorov flow data."""
        import numpy as np

        # Try to load from common formats
        data_file = self.data_path / f"{self.split}.pt"
        if data_file.exists():
            data = torch.load(data_file)
            if isinstance(data, dict):
                u = data.get("u", data.get("velocity_x"))
                v = data.get("v", data.get("velocity_y"))
                # Stack u and v, pad with zeros for 3rd channel
                self.data = torch.stack([u, v, torch.zeros_like(u)], dim=2)  # (N, T, 3, H, W)
            else:
                self.data = data
        else:
            # Try numpy format
            data_file = self.data_path / f"{self.split}.npy"
            if data_file.exists():
                data = np.load(data_file)
                self.data = torch.from_numpy(data).float()
            else:
                raise FileNotFoundError(f"Could not find data at {self.data_path}")

        if self.num_trajectories is not None:
            self.data = self.data[:self.num_trajectories]

        # Resize to 128x128 if needed
        N, T, C, H, W = self.data.shape
        if H != 128 or W != 128:
            self.data = F.interpolate(
                self.data.reshape(N * T, C, H, W),
                size=(128, 128),
                mode="bilinear",
                align_corners=False,
            ).reshape(N, T, C, 128, 128)

    def __len__(self):
        return len(self.data) * (self.data.shape[1] - 3)  # Sliding window of 4 frames

    def __getitem__(self, idx):
        traj_idx = idx // (self.data.shape[1] - 3)
        frame_idx = idx % (self.data.shape[1] - 3)
        return self.data[traj_idx, frame_idx:frame_idx + 4]  # (4, C, H, W)


def finetune_kolmogorov(args):
    """Finetune pretrained model on Kolmogorov turbulence."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained P2VAE
    vae = P2VAE_16M(in_channels=3, latent_channels=16)
    if args.vae_checkpoint:
        checkpoint = torch.load(args.vae_checkpoint, map_location="cpu")
        vae.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded P2VAE from {args.vae_checkpoint}")
    vae = vae.to(device)

    # Load pretrained FMT-B
    fmt = FMTBase(latent_channels=16, latent_size=16)
    if args.fmt_checkpoint:
        checkpoint = torch.load(args.fmt_checkpoint, map_location="cpu")
        fmt.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded FMT-B from {args.fmt_checkpoint}")
    fmt = fmt.to(device)

    # Dataset
    train_dataset = KolmogorovDataset(
        args.data_dir,
        split="train",
        num_trajectories=200,  # Few-shot: 200 trajectories
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Optimizer: optimize both VAE and FMT
    # Following REPA-E: stop-gradient after latent generation
    optimizer = AdamW(
        list(vae.parameters()) + list(fmt.parameters()),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    num_warmup_steps = int(0.1 * args.num_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.num_steps,
    )

    # Training loop
    step = 0
    vae.train()
    fmt.train()

    logger.info(f"Starting Kolmogorov finetuning for {args.num_steps} steps")
    logger.info(f"Training on {len(train_dataset)} samples from 200 trajectories")

    train_iter = iter(train_loader)

    while step < args.num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # batch: (B, 4, C, H, W)
        batch = batch.to(device)
        B, T, C, H, W = batch.shape

        # VAE loss (reconstruction)
        # Flatten all frames for VAE training
        frames_flat = batch.reshape(B * T, C, H, W)
        x_hat, vae_loss = vae(frames_flat)

        # CFM loss with stop-gradient on latents
        # Encode frames with stop-gradient (following REPA-E)
        with torch.no_grad():
            latents = []
            for t_idx in range(T):
                z = vae.get_latent(batch[:, t_idx], deterministic=True)
                latents.append(z.detach())  # stop-gradient

        # Sample t and k
        t_values = torch.rand(B, 4, device=device)
        k_values = torch.rand(B, 4, device=device)

        # Build noisy latents
        noisy_latents = []
        for s in range(4):
            if s < 3:
                x0 = latents[s]
                x1 = latents[s + 1]
            else:
                x0 = latents[s - 1]
                x1 = latents[s]

            t_s = t_values[:, s]
            k_s = k_values[:, s]
            x_noisy, _, _ = sample_interpolation(x0, x1, t_s, k_s)
            noisy_latents.append(x_noisy)

        # FMT forward pass
        t_target = t_values[:, -1]
        velocity_pred, _ = fmt(noisy_latents, t_target, t_all=t_values)

        # CFM loss
        x0_target = latents[-2]
        x1_target = latents[-1]
        x_noisy_target = noisy_latents[-1]  # Use same noisy state as model input
        cfm_loss = flow_marching_loss(velocity_pred, x_noisy_target, x1_target, t_target)

        # Total loss: L = L_CFM + lambda_VAE * L_VAE
        total_loss = cfm_loss + args.lambda_vae * vae_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(vae.parameters()) + list(fmt.parameters()), 1.0
        )
        optimizer.step()
        scheduler.step()

        step += 1

        if step % args.log_every == 0:
            logger.info(
                f"Step {step}/{args.num_steps}, "
                f"Total: {total_loss.item():.6f}, "
                f"CFM: {cfm_loss.item():.6f}, "
                f"VAE: {vae_loss.item():.6f}"
            )

    # Save finetuned model
    save_path = Path(args.output_dir) / "kolmogorov_finetuned.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "vae_state_dict": vae.state_dict(),
        "fmt_state_dict": fmt.state_dict(),
        "args": vars(args),
    }, save_path)
    logger.info(f"Saved finetuned model to {save_path}")

    return vae, fmt


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune on Kolmogorov turbulence")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Kolmogorov dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--vae_checkpoint", type=str, default=None, help="Path to P2VAE checkpoint")
    parser.add_argument("--fmt_checkpoint", type=str, default=None, help="Path to FMT-B checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_steps", type=int, default=5_000, help="Number of finetuning steps")
    parser.add_argument("--lambda_vae", type=float, default=1.0, help="VAE loss weight")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--log_every", type=int, default=100, help="Log every N steps")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    finetune_kolmogorov(args)
