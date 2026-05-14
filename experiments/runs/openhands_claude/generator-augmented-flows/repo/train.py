"""
Training script for consistency models with generator-augmented flows.

Implements Algorithm 1 from Issenhuth et al. (2024):
- iCT-IC: standard improved consistency training with independent coupling
- iCT-OT: iCT with minibatch optimal transport coupling
- iCT-GC (μ=0.5): iCT with generator-augmented coupling and joint learning

Also supports:
- Pre-trained predictor experiment (Section 5.1): use a frozen pre-trained model as endpoint predictor
- ECT setting (Section 5.3): fine-tune from a pre-trained diffusion model
"""

import argparse
import copy
import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.model import ConsistencyModel, build_consistency_model
from src.schedules import NoiseSchedule, TimestepSchedule, TimestepSampler, LossWeighting
from src.coupling import IndependentCoupling, BatchOTCoupling, JointCoupling
from src.losses import JointGCLoss, pseudo_huber_distance, get_pseudo_huber_c
from src.data import get_dataset, get_dataloader, InfiniteDataLoader
from src.metrics import EvaluationMetrics


def get_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Build the Lion optimizer as used in the paper (Chen et al., 2023)."""
    try:
        from lion_pytorch import Lion
        optimizer = Lion(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg.get("weight_decay", 0.0),
        )
    except ImportError:
        print("lion-pytorch not found, falling back to AdamW")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg.get("weight_decay", 0.0),
        )
    return optimizer


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999):
    """Update exponential moving average of model parameters."""
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def save_checkpoint(
    path: str,
    step: int,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: Optional[dict] = None,
):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    """Load training checkpoint, return the step number."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    ema_model.load_state_dict(checkpoint["ema_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("step", 0)


def load_predictor(path: str, cfg: dict, device: torch.device) -> ConsistencyModel:
    """
    Load a pre-trained consistency model to use as a frozen endpoint predictor.
    Used in the pre-trained predictor experiment (Section 5.1).
    """
    predictor = build_consistency_model(cfg).to(device)
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("ema_state_dict", checkpoint.get("model_state_dict"))
    predictor.load_state_dict(state_dict)
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad_(False)
    print(f"Loaded frozen predictor from {path}")
    return predictor


def train(cfg: dict, args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Override config with command-line args
    if args.coupling:
        cfg["coupling"] = args.coupling
    if args.mu is not None:
        cfg["mu"] = args.mu
    if args.lr is not None:
        cfg["learning_rate"] = args.lr

    coupling_type = cfg.get("coupling", "gc")
    mu = cfg.get("mu", 0.5)

    # Build model
    model = build_consistency_model(cfg).to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = get_optimizer(model, cfg)

    # Schedules
    noise_schedule = NoiseSchedule(
        sigma_min=cfg["sigma_min"],
        sigma_max=cfg["sigma_max"],
        rho=cfg.get("rho", 7.0),
    )
    timestep_schedule = TimestepSchedule(
        total_steps=cfg["training_steps"],
        s0=cfg.get("s0", 10),
        s1=cfg.get("s1", 1280),
    )
    timestep_sampler = TimestepSampler(
        p_mean=cfg.get("p_mean", -1.1),
        p_std=cfg.get("p_std", 2.0),
    )

    # Loss function with pseudo-Huber distance
    c = get_pseudo_huber_c(cfg["in_channels"], cfg["image_resolution"])
    distance_fn = lambda x, y: pseudo_huber_distance(x, y, c=c)
    loss_fn = JointGCLoss(mu=mu, distance_fn=distance_fn)

    # Couplings
    ic_coupling = IndependentCoupling()
    ot_coupling = None
    joint_coupling = None

    if coupling_type == "ot":
        ot_coupling = BatchOTCoupling(reg=cfg.get("ot_reg", 0.05))
    elif coupling_type == "gc":
        joint_coupling = JointCoupling(mu=mu)

    # Pre-trained predictor (Section 5.1): use a frozen model as endpoint predictor
    # When specified, overrides the EMA model for GC endpoint prediction
    frozen_predictor = None
    if args.predictor_checkpoint:
        frozen_predictor = load_predictor(args.predictor_checkpoint, cfg, device)
        print(f"Using frozen predictor for GC endpoint prediction (Section 5.1 experiment)")

    # Dataset
    dataset = get_dataset(
        name=cfg["dataset"],
        root=cfg["data_root"],
        resolution=cfg["image_resolution"],
        train=True,
    )
    dataloader = get_dataloader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 4),
    )
    data_iter = InfiniteDataLoader(dataloader)

    # Evaluation dataset (use training set for FID as per standard practice)
    eval_dataloader = get_dataloader(
        dataset,
        batch_size=cfg.get("eval_batch_size", 256),
        shuffle=False,
        drop_last=False,
    )
    evaluator = EvaluationMetrics(device=device, num_samples=cfg.get("eval_samples", 50000))

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, ema_model, optimizer)
        print(f"Resumed from step {start_step}")

    output_dir = cfg.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    log_interval = cfg.get("log_interval", 100)
    eval_interval = cfg.get("eval_interval", 10000)
    save_interval = cfg.get("save_interval", 10000)

    print(f"Training {coupling_type.upper()} consistency model for {cfg['training_steps']} steps")
    print(f"Coupling: {coupling_type}, μ={mu}")

    # Cache sigmas to avoid recomputing when N doesn't change
    _cached_N = -1
    _cached_sigmas = None

    model.train()
    t0 = time.time()

    for step in range(start_step, cfg["training_steps"]):
        # Get current number of timesteps N(k) from exponential schedule
        N = timestep_schedule.get_N(step)

        # Recompute sigmas only when N changes
        if N != _cached_N:
            _cached_sigmas = noise_schedule.get_sigmas(N, device=device)
            _cached_N = N
        sigmas = _cached_sigmas

        # Sample timestep indices according to p(σ_i) distribution
        indices = timestep_sampler.sample_indices(sigmas, cfg["batch_size"], device)

        # σ_{t_i} (lower) and σ_{t_{i+1}} (upper) for each sample
        sigma_lower = sigmas[indices]
        sigma_upper = sigmas[indices + 1]

        # Loss weights λ(σ_{t_i}) = 1 / (σ_{t_{i+1}} - σ_{t_i})
        all_weights = LossWeighting.get_weights(sigmas)
        loss_weights = all_weights[indices].to(device)

        # Sample data and noise
        x_star = next(data_iter).to(device)
        z = torch.randn_like(x_star)

        # Determine endpoint predictor for GC:
        # - frozen_predictor: pre-trained model (Section 5.1 experiment)
        # - ema_model: current EMA model (default joint learning, Algorithm 1)
        endpoint_predictor = frozen_predictor if frozen_predictor is not None else ema_model

        # Construct training pairs based on coupling type
        if coupling_type == "ic":
            x_lower_pair, x_upper_pair = ic_coupling.construct_pairs(
                x_star, z, sigma_lower, sigma_upper
            )
        elif coupling_type == "ot":
            x_lower_pair, x_upper_pair = ot_coupling.construct_pairs(
                x_star, z, sigma_lower, sigma_upper
            )
        elif coupling_type == "gc":
            # Algorithm 1: joint learning with IC and GC trajectories
            # m ~ Binomial(μ, batch_size) selects which samples use GC vs IC
            x_lower_pair, x_upper_pair = joint_coupling.construct_pairs(
                x_star, z, sigma_lower, sigma_upper, model=endpoint_predictor
            )
        else:
            raise ValueError(f"Unknown coupling type: {coupling_type}")

        # Compute consistency loss
        loss = loss_fn(
            model=model,
            x_lower=x_lower_pair,
            x_upper=x_upper_pair,
            sigma_lower=sigma_lower,
            sigma_upper=sigma_upper,
            loss_weights=loss_weights,
        )

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update EMA (used for GC endpoint prediction and final evaluation)
        ema_decay = cfg.get("ema_decay", 0.9999)
        update_ema(ema_model, model, decay=ema_decay)

        # Logging
        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t0
            steps_per_sec = log_interval / elapsed
            print(
                f"Step {step + 1}/{cfg['training_steps']} | "
                f"Loss: {loss.item():.4f} | "
                f"N: {N} | "
                f"Steps/s: {steps_per_sec:.1f}"
            )
            t0 = time.time()

        # Evaluation
        if (step + 1) % eval_interval == 0:
            print(f"Evaluating at step {step + 1}...")
            metrics = evaluator.evaluate(
                ema_model,
                eval_dataloader,
                sigma_max=cfg["sigma_max"],
                num_steps=1,
            )
            print(
                f"Step {step + 1} | "
                f"FID: {metrics['fid']:.2f} | "
                f"KID: {metrics['kid_mean'] * 100:.2f}±{metrics['kid_std'] * 100:.2f} (×10²) | "
                f"IS: {metrics['is_mean']:.2f}±{metrics['is_std']:.2f}"
            )
            model.train()

            ckpt_path = os.path.join(output_dir, f"checkpoint_{step + 1}.pt")
            save_checkpoint(ckpt_path, step + 1, model, ema_model, optimizer, metrics)

        elif (step + 1) % save_interval == 0:
            ckpt_path = os.path.join(output_dir, f"checkpoint_{step + 1}.pt")
            save_checkpoint(ckpt_path, step + 1, model, ema_model, optimizer)

    # Final save
    final_path = os.path.join(output_dir, "checkpoint_final.pt")
    save_checkpoint(final_path, cfg["training_steps"], model, ema_model, optimizer)
    print(f"Training complete. Final checkpoint saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train consistency models with generator-augmented flows")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument(
        "--coupling", type=str, choices=["ic", "ot", "gc"], default=None,
        help="Coupling type: ic (independent), ot (optimal transport), gc (generator-augmented)"
    )
    parser.add_argument(
        "--mu", type=float, default=None,
        help="Joint learning factor μ (0=IC only, 1=GC only, 0.5=default for iCT, 0.3 for ECT)"
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory override")
    parser.add_argument(
        "--predictor_checkpoint", type=str, default=None,
        help="Path to pre-trained IC model to use as frozen endpoint predictor (Section 5.1)"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    train(cfg, args)


if __name__ == "__main__":
    main()
