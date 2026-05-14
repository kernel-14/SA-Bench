"""
Training loops for consistency models.

Implements:
1. Standard CT training (IC or batch-OT)
2. GC training with joint learning (Algorithm 1 from the paper)
3. EMA tracking
4. Training utilities
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Callable
from collections import OrderedDict
from copy import deepcopy
import time

from .scheduling import (
    noise_schedule_karras,
    weighting_function,
    discretization_schedule,
    sample_timesteps,
    get_sigmas_for_indices,
)
from .losses import (
    consistency_training_loss,
    gc_consistency_loss,
    joint_gc_loss,
    pseudo_huber_loss,
    get_distance_fn,
)


class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Replace model params with EMA shadow for inference."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original model params."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class ConsistencyTrainingConfig:
    """Configuration for consistency model training."""

    def __init__(
        self,
        # Noise schedule
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,

        # Discretization schedule
        s0: int = 10,
        s1: int = 1280,

        # Timestep distribution
        P_mean: float = -1.1,
        P_std: float = 2.0,

        # Training
        batch_size: int = 512,
        total_steps: int = 100_000,
        learning_rate: float = 1e-4,
        optimizer: str = "lion",
        ema_decay: float = 0.999,

        # Coupling
        coupling: str = "ic",  # "ic", "ot", "gc"
        gc_mu: float = 0.5,  # joint learning factor for GC
        ot_solver: str = "sinkhorn",

        # Model
        sigma_data: float = 0.5,

        # Distance function
        distance_fn_name: str = "pseudo_huber",
        distance_fn_kwargs: Dict = None,

        # Logging
        log_every: int = 100,
        eval_every: int = 5000,
        use_amp: bool = False,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.s0 = s0
        self.s1 = s1
        self.P_mean = P_mean
        self.P_std = P_std
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.ema_decay = ema_decay
        self.coupling = coupling
        self.gc_mu = gc_mu
        self.ot_solver = ot_solver
        self.sigma_data = sigma_data
        self.distance_fn_name = distance_fn_name
        self.distance_fn_kwargs = distance_fn_kwargs or {}
        self.log_every = log_every
        self.eval_every = eval_every
        self.use_amp = use_amp


def train_consistency_model(
    consistency_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: ConsistencyTrainingConfig,
    device: torch.device = None,
    callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Train a consistency model using the IC or OT baseline.

    Args:
        consistency_model: The consistency model f_θ
        dataloader: DataLoader providing batches of images
        config: Training configuration
        device: Torch device
        callback: Optional callback(step, loss, model) for logging

    Returns:
        Dictionary of training statistics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = consistency_model.to(device)

    # Optimizer
    if config.optimizer == "lion":
        opt = LionOptimizer(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # EMA
    ema = EMA(model, decay=config.ema_decay)

    # Distance function
    distance_fn = get_distance_fn(config.distance_fn_name, **config.distance_fn_kwargs)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if config.use_amp else None

    stats = {"losses": [], "step_times": []}
    step = 0
    epoch = 0

    while step < config.total_steps:
        epoch += 1
        for batch in dataloader:
            if step >= config.total_steps:
                break

            start_time = time.time()

            x_star = batch[0] if isinstance(batch, (list, tuple)) else batch
            x_star = x_star.to(device)
            B = x_star.shape[0]

            # Sample noise
            z = torch.randn_like(x_star)

            # Get current number of timesteps
            N = discretization_schedule(step, config.total_steps, config.s0, config.s1)
            sigmas = noise_schedule_karras(N, config.sigma_min, config.sigma_max, config.rho).to(device)

            # Sample timestep indices
            indices = sample_timesteps(B, sigmas, config.P_mean, config.P_std, device)
            sigma_i, sigma_next = get_sigmas_for_indices(sigmas, indices)

            # Get weights for the selected indices
            all_weights = weighting_function(sigmas)
            lambda_weight = all_weights[indices]

            # Compute loss based on coupling
            if config.coupling == "gc":
                # Use GC loss with joint learning
                loss = joint_gc_loss(
                    model, x_star, z, sigma_i, sigma_next,
                    lambda_weight, mu=config.gc_mu, distance_fn=distance_fn,
                )
            elif config.coupling == "ot":
                loss = consistency_training_loss(
                    model, x_star, z, sigma_i, sigma_next,
                    lambda_weight, distance_fn=distance_fn,
                    use_ot=True, ot_solver=config.ot_solver,
                )
            else:  # IC
                loss = consistency_training_loss(
                    model, x_star, z, sigma_i, sigma_next,
                    lambda_weight, distance_fn=distance_fn,
                )

            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            opt.zero_grad()
            ema.update()

            step_time = time.time() - start_time

            stats["losses"].append(loss.item())
            stats["step_times"].append(step_time)

            if step % config.log_every == 0:
                print(f"Step {step:6d}/{config.total_steps} | Loss: {loss.item():.6f} | "
                      f"N: {N} | Time: {step_time:.3f}s")

            if callback is not None:
                callback(step, loss.item(), model)

            step += 1

    # Apply EMA for final model
    ema.apply_shadow()

    return stats


class LionOptimizer(torch.optim.Optimizer):
    """
    Lion optimizer (Chen et al., 2023) as used in the paper.

    Implementation adapted from the paper's reference:
    https://github.com/lucidrains/lion-pytorch
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
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
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                # Weight decay
                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                # Lion update
                update = exp_avg.clone().lerp_(grad, 1 - beta1)
                update.sign_()  # sign(grad)
                p.add_(update, alpha=-lr)

                # Update exponential moving average
                exp_avg.lerp_(grad, 1 - beta2)

        return loss


def train_gc_joint(
    consistency_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: ConsistencyTrainingConfig = None,
    device: torch.device = None,
    mu: float = 0.5,
    **kwargs,
) -> Dict[str, Any]:
    """
    Train a consistency model with the joint GC learning strategy.

    This is the main training method from Algorithm 1 in the paper.

    Args:
        consistency_model: The consistency model
        dataloader: DataLoader
        config: Training configuration
        device: Device
        mu: Joint learning factor
        **kwargs: Override config values

    Returns:
        Training stats
    """
    if config is None:
        config = ConsistencyTrainingConfig()

    # Override with kwargs
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)

    config.coupling = "gc"
    config.gc_mu = mu

    return train_consistency_model(
        consistency_model, dataloader, config, device
    )
