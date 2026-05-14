"""
Easy Consistency Tuning (ECT) with Generator-Augmented Flows.

Based on Geng et al. (2024) "Consistency Models Made Easy"
and the ECT experiments in Section 5.3 of Issenhuth et al.

ECT fine-tunes a consistency model from a pre-trained diffusion model
using the consistency loss with a fixed small number of timesteps.
GC trajectories can be used in place of IC trajectories during ECT fine-tuning.
"""
import os
import math
import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ConsistencyModel, EMAHelper
from coupling import construct_gc_pairs
from loss import joint_gc_loss, weighting_fn, distance_fn
from data import create_dataloader
from metrics import evaluate_model
from config import Config, DATASET_CONFIGS


class ECTConfig:
    """ECT-specific hyperparameters from Geng et al. (2024)."""

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        num_timesteps: int = 18,  # Fixed small number of timesteps in ECT
        lr: float = 1e-5,
        batch_size: int = 256,
        total_steps_short: int = 4_000,  # ~1 GPU-hour
        total_steps_long: int = 100_000,
        mu: float = 0.3,  # Optimal for ECT-GC from paper
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.num_timesteps = num_timesteps
        self.lr = lr
        self.batch_size = batch_size
        self.total_steps_short = total_steps_short
        self.total_steps_long = total_steps_long
        self.mu = mu


def get_ect_noise_schedule(
    sigma_min: float,
    sigma_max: float,
    rho: float,
    N: int,
    device: torch.device,
) -> torch.Tensor:
    """Karras noise schedule."""
    rho_inv = 1.0 / rho
    indices = torch.arange(N + 1, device=device, dtype=torch.float32)
    sigmas = (sigma_min ** rho_inv + indices / N * (sigma_max ** rho_inv - sigma_min ** rho_inv)) ** rho
    return sigmas


def ect_train_step(
    model: ConsistencyModel,
    x_star: torch.Tensor,
    z: torch.Tensor,
    sigma_ti: torch.Tensor,
    sigma_ti_plus_1: torch.Tensor,
    mu: float,
    coupling: str = "gc",
    ema_helper: Optional[EMAHelper] = None,
    use_ema: bool = False,
):
    """ECT training step with GC coupling."""
    B = x_star.shape[0]
    device = x_star.device
    loss_type = "pseudo_huber"

    if coupling == "ic":
        x_ti = x_star + sigma_ti[:, None, None, None] * z
        x_ti_plus_1 = x_star + sigma_ti_plus_1[:, None, None, None] * z
        mask = torch.zeros(B, device=device)
    elif coupling == "gc":
        mask = torch.rand(B, device=device) < mu
        x_ti, x_ti_plus_1, _, _ = construct_gc_pairs(
            x_star=x_star, z=z, sigma_ti=sigma_ti, sigma_ti_plus_1=sigma_ti_plus_1,
            consistency_model=model, mask=mask,
            use_ema=use_ema, ema_helper=ema_helper,
        )
    else:
        raise ValueError(f"Unknown coupling: {coupling}")

    return joint_gc_loss(model, x_ti, x_ti_plus_1, sigma_ti, sigma_ti_plus_1, mask, loss_type)


def train_ect(
    model: ConsistencyModel,
    dataloader: DataLoader,
    eval_loader: DataLoader,
    ect_config: ECTConfig,
    total_steps: int,
    coupling: str = "gc",
    output_dir: str = "./ect_output",
    device: torch.device = torch.device("cuda"),
):
    """Run ECT fine-tuning."""
    os.makedirs(output_dir, exist_ok=True)

    # EMA on fine-tuned model
    ema_helper = EMAHelper(model, mu=0.9999)

    optimizer = optim.Adam(model.parameters(), lr=ect_config.lr)

    N = ect_config.num_timesteps
    sigmas = get_ect_noise_schedule(
        ect_config.sigma_min, ect_config.sigma_max, ect_config.rho, N, device
    )

    # Uniform timestep sampling for ECT
    probs = torch.ones(N, device=device) / N

    pbar = tqdm(total=total_steps, desc=f"ECT-{coupling.upper()}")
    train_iter = iter(dataloader)
    global_step = 0

    while global_step < total_steps:
        try:
            x_star, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(dataloader)
            x_star, _ = next(train_iter)

        x_star = x_star.to(device)
        B = x_star.shape[0]

        # Sample timestep
        i_indices = torch.multinomial(probs, num_samples=B, replacement=True)
        sigma_ti = sigmas[i_indices]
        sigma_ti_plus_1 = sigmas[i_indices + 1]

        z = torch.randn_like(x_star)

        loss = ect_train_step(
            model, x_star, z, sigma_ti, sigma_ti_plus_1,
            mu=ect_config.mu, coupling=coupling,
            ema_helper=ema_helper, use_ema=True,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ema_helper.update(model)

        global_step += 1
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Evaluation at checkpoints
        if global_step % 2000 == 0 or global_step == total_steps:
            ema_helper.store(model)
            ema_helper.apply_to(model)

            metrics = evaluate_model(
                model, eval_loader,
                num_samples=min(50000, len(eval_loader.dataset)),
                batch_size=256, device=device,
            )
            print(f"\nECT Step {global_step}: FID={metrics['fid']:.4f}")

            torch.save({
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
            }, os.path.join(output_dir, f"ect_checkpoint_{global_step}.pt"))

            ema_helper.restore(model)

    pbar.close()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--coupling", type=str, default="gc", choices=["ic", "gc"])
    parser.add_argument("--mu", type=float, default=0.3)
    parser.add_argument("--pretrained_ckpt", type=str, required=True,
                        help="Path to pre-trained diffusion model checkpoint")
    parser.add_argument("--training_length", type=str, default="short",
                        choices=["short", "long"])
    parser.add_argument("--output_dir", type=str, default="./ect_output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = DATASET_CONFIGS.get(args.dataset, DATASET_CONFIGS["cifar10"])
    image_size = config.training.image_size

    # Create consistency model (to be initialized from pre-trained diffusion)
    model = ConsistencyModel(
        in_channels=3, out_channels=3,
        model_channels=config.model.model_channels,
        num_blocks=config.model.num_blocks,
        channel_mult=config.model.channel_mult,
        attn_resolutions=config.model.attn_resolutions,
        dropout=config.model.dropout,
        embedding_type=config.model.embedding_type,
        sigma_data=config.model.sigma_data,
        img_resolution=image_size,
    ).to(device)

    # Load pre-trained weights (ECT initializes from diffusion model)
    if os.path.exists(args.pretrained_ckpt):
        ckpt = torch.load(args.pretrained_ckpt, map_location=device)
        # ECT initialization: load teacher diffusion model weights
        # The consistency model needs special initialization from the diffusion model
        # For simplicity, load directly if weights are compatible
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"Loaded pre-trained checkpoint from {args.pretrained_ckpt}")

    ect_config = ECTConfig(mu=args.mu)

    dataloader = create_dataloader(
        name=args.dataset, image_size=image_size,
        batch_size=ect_config.batch_size,
    )
    eval_loader = create_dataloader(
        name=args.dataset, image_size=image_size,
        batch_size=256, shuffle=False, drop_last=False, train=True,
    )

    ect_config = ECTConfig(mu=args.mu)
    total_steps = (
        ect_config.total_steps_short if args.training_length == "short"
        else ect_config.total_steps_long
    )

    train_ect(
        model, dataloader, eval_loader, ect_config,
        total_steps=total_steps, coupling=args.coupling,
        output_dir=args.output_dir, device=device,
    )


if __name__ == "__main__":
    main()
