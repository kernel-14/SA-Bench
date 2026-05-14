"""
Training script for consistency models with Generator-Augmented Flows.

Implements:
- Standard iCT (Independent Coupling)
- iCT with batch-OT coupling
- iCT with Generator-Augmented Coupling (GC) and joint learning

Based on Song & Dhariwal (2024) "Improved Techniques for Training Consistency Models"
and Issenhuth et al. "Improving Consistency Models with Generator-Augmented Flows".
"""
import os
import sys
import math
import copy
import argparse
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config import Config, DATASET_CONFIGS
from model import ConsistencyModel, EMAHelper
from coupling import (
    independent_coupling,
    batch_ot_coupling,
    construct_gc_pairs,
)
from loss import joint_gc_loss
from data import create_dataloader
from metrics import evaluate_model


def get_noise_schedule(
    sigma_min: float,
    sigma_max: float,
    rho: float,
    num_timesteps: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Karras et al. (2022) noise schedule:
    sigma_i = (sigma_min^{1/rho} + i/N * (sigma_max^{1/rho} - sigma_min^{1/rho}))^rho
    """
    rho_inv = 1.0 / rho
    indices = torch.arange(num_timesteps + 1, device=device, dtype=torch.float32)
    sigmas = (sigma_min ** rho_inv + indices / num_timesteps * (sigma_max ** rho_inv - sigma_min ** rho_inv)) ** rho
    return sigmas


def get_discretization_schedule(
    step: int,
    total_steps: int,
    s0: int,
    s1: int,
) -> int:
    """
    Exponential discretization schedule from Song & Dhariwal (2024):
    N(k) = min(s0 * 2^{floor(k / K')}, s1) + 1
    where K' = floor(K / (log_2(s1/s0) + 1))
    """
    K_prime = math.floor(total_steps / (math.log2(s1 / s0) + 1))
    N = min(s0 * 2 ** math.floor(step / K_prime), s1) + 1
    return N


def get_timestep_sampling_probs(
    sigmas: torch.Tensor,
    p_mean: float = -1.1,
    p_std: float = 2.0,
) -> torch.Tensor:
    """
    Discrete timestep sampling distribution (Gaussian in log-space).
    p(sigma_i) ∝ erf(log(sigma_{i+1}) - p_mean) / (sqrt(2) * p_std)) - erf(log(sigma_i) - p_mean) / (sqrt(2) * p_std))
    """
    log_sigmas = sigmas.log()
    cdf = 0.5 * (1.0 + torch.erf((log_sigmas - p_mean) / (p_std * math.sqrt(2))))
    probs = cdf[1:] - cdf[:-1]
    probs = probs / probs.sum()
    return probs


class LionOptimizer(torch.optim.Optimizer):
    """
    Lion optimizer from Chen et al. (2023).
    Implementation based on https://github.com/lucidrains/lion-pytorch.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0 or 1: {betas}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                update = exp_avg.clone()
                update.mul_(beta2).add_(grad, alpha=1.0 - beta2)
                update.sign_()

                p.add_(update, alpha=-lr)
                if wd > 0:
                    p.add_(p, alpha=-lr * wd)

        return loss


def get_optimizer(
    model: nn.Module,
    lr: float,
    optimizer_type: str = "lion",
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Get optimizer instance."""
    if optimizer_type == "lion":
        return LionOptimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def train_step_ic(
    model: ConsistencyModel,
    x_star: torch.Tensor,
    z: torch.Tensor,
    sigma_ti: torch.Tensor,
    sigma_ti_plus_1: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    """Training step with Independent Coupling (IC)."""
    x_ti = x_star + sigma_ti[:, None, None, None] * z
    x_ti_plus_1 = x_star + sigma_ti_plus_1[:, None, None, None] * z

    return joint_gc_loss(
        model, x_ti, x_ti_plus_1, sigma_ti, sigma_ti_plus_1,
        mask=torch.zeros(x_star.shape[0], device=x_star.device),
        loss_type=loss_type,
    )


def train_step_gc(
    model: ConsistencyModel,
    x_star: torch.Tensor,
    z: torch.Tensor,
    sigma_ti: torch.Tensor,
    sigma_ti_plus_1: torch.Tensor,
    mu: float,
    loss_type: str,
    ema_helper: Optional[EMAHelper] = None,
    use_ema: bool = False,
) -> torch.Tensor:
    """Training step with Generator-Augmented Coupling (GC) and joint learning."""
    B = x_star.shape[0]
    device = x_star.device

    # Generate mask: each sample uses GC with probability mu
    mask = torch.rand(B, device=device) < mu

    # Construct mixed pairs
    x_ti, x_ti_plus_1, _, _ = construct_gc_pairs(
        x_star=x_star,
        z=z,
        sigma_ti=sigma_ti,
        sigma_ti_plus_1=sigma_ti_plus_1,
        consistency_model=model,
        mask=mask,
        use_ema=use_ema,
        ema_helper=ema_helper,
    )

    return joint_gc_loss(
        model, x_ti, x_ti_plus_1, sigma_ti, sigma_ti_plus_1,
        mask=mask, loss_type=loss_type,
    )


def train_step_ot(
    model: ConsistencyModel,
    x_star: torch.Tensor,
    z: torch.Tensor,
    sigma_ti: torch.Tensor,
    sigma_ti_plus_1: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    """Training step with batch-OT coupling."""
    x_star_perm, z_perm = batch_ot_coupling(x_star, z)

    x_ti = x_star_perm + sigma_ti[:, None, None, None] * z_perm
    x_ti_plus_1 = x_star_perm + sigma_ti_plus_1[:, None, None, None] * z_perm

    return joint_gc_loss(
        model, x_ti, x_ti_plus_1, sigma_ti, sigma_ti_plus_1,
        mask=torch.zeros(x_star.shape[0], device=x_star.device),
        loss_type=loss_type,
    )


def train(config: Config, output_dir: str = "./output"):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Set seed
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)

    # Create dataloader
    train_loader = create_dataloader(
        name=config.training.dataset,
        image_size=config.training.image_size,
        batch_size=config.training.batch_size,
    )

    # For FID evaluation
    eval_loader = create_dataloader(
        name=config.training.dataset,
        image_size=config.training.image_size,
        batch_size=256,
        shuffle=False,
        drop_last=False,
        train=True,  # Use training set for FID reference as in paper
    )

    # Create model
    model = ConsistencyModel(
        in_channels=3,
        out_channels=3,
        model_channels=config.model.model_channels,
        num_blocks=config.model.num_blocks,
        channel_mult=config.model.channel_mult,
        attn_resolutions=config.model.attn_resolutions,
        dropout=config.model.dropout,
        embedding_type=config.model.embedding_type,
        sigma_data=config.model.sigma_data,
        img_resolution=config.training.image_size,
    ).to(device)

    # Create EMA
    ema_helper = EMAHelper(model, mu=config.model.ema_rate) if config.model.use_ema else None

    # Create optimizer
    optimizer = get_optimizer(
        model,
        lr=config.training.learning_rate,
        optimizer_type=config.training.optimizer,
        weight_decay=config.training.weight_decay,
    )

    # Mixed precision
    scaler = GradScaler() if config.training.mixed_precision else None

    # Training state
    global_step = 0
    total_steps = config.training.total_steps
    s0, s1 = config.schedule.s0, config.schedule.s1
    mu = config.training.mu
    loss_type = config.training.loss_type
    coupling = config.training.coupling

    pbar = tqdm(total=total_steps, desc="Training")
    train_iter = iter(train_loader)

    while global_step < total_steps:
        try:
            x_star, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x_star, _ = next(train_iter)

        x_star = x_star.to(device)
        B = x_star.shape[0]

        # Get number of timesteps from discretization schedule
        N = get_discretization_schedule(global_step, total_steps, s0, s1)

        # Get noise schedule
        sigmas = get_noise_schedule(
            config.schedule.sigma_min,
            config.schedule.sigma_max,
            config.schedule.rho,
            N,
            device,
        )

        # Sample timestep index i
        probs = get_timestep_sampling_probs(sigmas, config.schedule.p_mean, config.schedule.p_std)
        i_indices = torch.multinomial(probs, num_samples=B, replacement=True)

        # Get sigma_ti and sigma_{t+1}
        sigma_ti = sigmas[i_indices]
        sigma_ti_plus_1 = sigmas[i_indices + 1]

        # Sample noise
        z = torch.randn_like(x_star)

        # Forward pass based on coupling type
        if coupling == "ic":
            loss = train_step_ic(model, x_star, z, sigma_ti, sigma_ti_plus_1, loss_type)
        elif coupling == "ot":
            loss = train_step_ot(model, x_star, z, sigma_ti, sigma_ti_plus_1, loss_type)
        elif coupling == "gc":
            loss = train_step_gc(
                model, x_star, z, sigma_ti, sigma_ti_plus_1,
                mu=mu, loss_type=loss_type,
                ema_helper=ema_helper,
                use_ema=config.training.use_ema_target,
            )
        else:
            raise ValueError(f"Unknown coupling: {coupling}")

        # Backward pass
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if config.training.gradient_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.training.gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            optimizer.step()

        # Update EMA
        if ema_helper is not None:
            ema_helper.update(model)

        global_step += 1
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "N": N})

        # Periodic FID evaluation
        if global_step % 10000 == 0 or global_step == total_steps:
            if ema_helper is not None:
                ema_helper.store(model)
                ema_helper.apply_to(model)

            metrics = evaluate_model(
                model,
                eval_loader,
                num_samples=config.training.num_fid_samples,
                batch_size=256,
                device=device,
            )
            print(f"\nStep {global_step}: FID={metrics['fid']:.4f}, KID={metrics['kid']:.4f}, IS={metrics['is_mean']:.4f}")

            # Save checkpoint
            checkpoint = {
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            }
            if ema_helper is not None:
                checkpoint["ema"] = ema_helper.shadow
            torch.save(checkpoint, os.path.join(output_dir, f"checkpoint_{global_step}.pt"))

            if ema_helper is not None:
                ema_helper.restore(model)

    pbar.close()

    # Final evaluation
    if ema_helper is not None:
        ema_helper.store(model)
        ema_helper.apply_to(model)

    final_metrics = evaluate_model(
        model,
        eval_loader,
        num_samples=config.training.num_fid_samples,
        batch_size=256,
        device=device,
    )

    print(f"\nFinal results: FID={final_metrics['fid']:.4f}, KID={final_metrics['kid']:.4f}, IS={final_metrics['is_mean']:.4f}")

    # Save final model
    torch.save({"model_state_dict": model.state_dict(), "metrics": final_metrics}, os.path.join(output_dir, "model_final.pt"))

    return final_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=DATASET_CONFIGS.keys())
    parser.add_argument("--coupling", type=str, default="gc", choices=["ic", "ot", "gc"])
    parser.add_argument("--mu", type=float, default=0.5, help="Joint learning parameter for GC")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--total_steps", type=int, default=None, help="Override total steps")
    parser.add_argument("--loss_type", type=str, default="pseudo_huber", choices=["pseudo_huber", "l2"])
    parser.add_argument("--dropout", type=float, default=None, help="Override dropout")
    parser.add_argument("--use_ema_target", action="store_true", help="Use EMA for GC endpoint prediction")
    parser.add_argument("--no_ema_target", action="store_true", help="Do not use EMA for GC endpoint prediction")
    args = parser.parse_args()

    # Load dataset config
    config = DATASET_CONFIGS.get(args.dataset, DATASET_CONFIGS["cifar10"])
    config.training.coupling = args.coupling
    config.training.mu = args.mu
    config.training.loss_type = args.loss_type

    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.total_steps is not None:
        config.training.total_steps = args.total_steps
    if args.dropout is not None:
        config.model.dropout = args.dropout
        config.training.use_dropout = args.dropout > 0

    if args.use_ema_target:
        config.training.use_ema_target = True
    if args.no_ema_target:
        config.training.use_ema_target = False

    print(f"Training config: {config}")
    train(config, args.output_dir)


if __name__ == "__main__":
    main()
