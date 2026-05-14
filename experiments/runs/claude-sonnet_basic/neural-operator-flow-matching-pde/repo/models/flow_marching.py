"""
Flow Marching Algorithm

Implements the core flow marching interpolation kernel and training objectives.

Key equations:
    x_t^k = mu_t + sigma_t * z
    mu_t = t*x1 + k*(1-t)*x0
    sigma_t = (1-t)*(1-k)
    z ~ N(0, I)

    Velocity: u_t^k = (x1 - x_t^k) / (1-t)

    Training objective (flow marching):
        L_FM = 0.5 * E[||(1-t)*g(x_t^k, t) - (x1 - x_t^k)||^2]

    Conditional flow marching:
        L_CFM = 0.5 * E[||(1-t)*g(x_t^k, t, h) - (x1 - x_t^k)||^2]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def sample_interpolation(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample intermediate state x_t^k using the location-scale interpolation kernel.

    x_t^k = mu_t + sigma_t * z
    mu_t = t*x1 + k*(1-t)*x0
    sigma_t = (1-t)*(1-k)
    z ~ N(0, I)

    Args:
        x0: (B, ...) current state
        x1: (B, ...) next state
        t: (B,) time values in [0, 1], sampled uniformly if None
        k: (B,) bridge parameter in [0, 1], sampled uniformly if None

    Returns:
        x_t_k: (B, ...) interpolated state
        t: (B,) time values used
        k: (B,) bridge parameters used
    """
    B = x0.shape[0]
    device = x0.device
    dtype = x0.dtype

    if t is None:
        t = torch.rand(B, device=device, dtype=dtype)
    if k is None:
        k = torch.rand(B, device=device, dtype=dtype)

    # Reshape for broadcasting
    shape = (B,) + (1,) * (x0.dim() - 1)
    t_b = t.reshape(shape)
    k_b = k.reshape(shape)

    # Compute mean and std
    mu_t = t_b * x1 + k_b * (1 - t_b) * x0
    sigma_t = (1 - t_b) * (1 - k_b)

    # Sample noise
    z = torch.randn_like(x0)

    # Interpolated state
    x_t_k = mu_t + sigma_t * z

    return x_t_k, t, k


def compute_velocity_target(
    x_t_k: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the velocity target for flow marching training.

    The k-free velocity target is:
        u_t^k = (x1 - x_t^k) / (1-t)

    Args:
        x_t_k: (B, ...) interpolated state
        x1: (B, ...) target state
        t: (B,) time values

    Returns:
        velocity: (B, ...) velocity target
    """
    B = x_t_k.shape[0]
    shape = (B,) + (1,) * (x_t_k.dim() - 1)
    t_b = t.reshape(shape)

    # Clamp to avoid division by zero near t=1
    denom = (1 - t_b).clamp(min=1e-6)
    return (x1 - x_t_k) / denom


def flow_marching_loss(
    predicted_velocity: torch.Tensor,
    x_t_k: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the flow marching training loss.

    L_FM = 0.5 * E[||(1-t)*g(x_t^k, t) - (x1 - x_t^k)||^2]

    The (1-t) preconditioning makes the loss numerically stable near t->1.

    Args:
        predicted_velocity: (B, ...) model's predicted velocity g(x_t^k, t)
        x_t_k: (B, ...) interpolated state
        x1: (B, ...) target state
        t: (B,) time values

    Returns:
        loss: scalar loss value
    """
    B = predicted_velocity.shape[0]
    shape = (B,) + (1,) * (predicted_velocity.dim() - 1)
    t_b = t.reshape(shape)

    # Target: (x1 - x_t^k)
    target = x1 - x_t_k

    # Preconditioned prediction: (1-t) * g
    pred_precond = (1 - t_b) * predicted_velocity

    loss = 0.5 * F.mse_loss(pred_precond, target, reduction="mean")
    return loss


def euler_step(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    t: float,
    dt: float,
) -> torch.Tensor:
    """
    Euler ODE step for flow marching inference.

    x_{t+dt} = x_t + dt * g(x_t, t)

    Args:
        x_t: (B, ...) current state
        velocity: (B, ...) predicted velocity
        t: current time
        dt: time step

    Returns:
        x_next: (B, ...) next state
    """
    return x_t + dt * velocity


@torch.no_grad()
def flow_marching_sample(
    model,
    frames_context: list,
    h_prev: Optional[torch.Tensor],
    device: torch.device,
    num_steps: int = 100,
    k_target: float = 0.0,
    batch_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate next state using Euler ODE sampler.

    Starting from x_t^k at t=0, integrate the velocity field to t=1.

    Args:
        model: FlowMarchingTransformer
        frames_context: list of 3 context frames (latent tensors)
        h_prev: (B, embed_dim) previous diffusion forcing state
        device: computation device
        num_steps: number of Euler steps (N=100 in paper)
        k_target: bridge parameter for target frame (0=pure noise, 1=deterministic)
        batch_size: batch size

    Returns:
        x1_pred: (B, C, H, W) predicted next state
        h_new: (B, embed_dim) updated diffusion forcing state
    """
    dt = 1.0 / num_steps

    # Initialize target frame
    # For k=0: start from pure noise
    # For k=1: start from x0 (deterministic)
    latent_channels = frames_context[0].shape[1]
    latent_size = frames_context[0].shape[2]

    if k_target < 1.0:
        # Start from noisy initialization
        x_t = torch.randn(batch_size, latent_channels, latent_size, latent_size, device=device)
        x_t = x_t * (1 - k_target)
        if k_target > 0:
            x_t = x_t + k_target * frames_context[-1]
    else:
        # Deterministic: start from last context frame
        x_t = frames_context[-1].clone()

    # Euler integration
    t_val = 0.0
    h_new = h_prev

    for step in range(num_steps):
        t_tensor = torch.full((batch_size,), t_val, device=device)

        # Build input frames: 3 context + current target
        input_frames = list(frames_context) + [x_t]

        # Get velocity prediction
        velocity, h_new = model(input_frames, t_tensor, h_new)

        # Euler step
        x_t = euler_step(x_t, velocity, t_val, dt)
        t_val += dt

    return x_t, h_new


class FlowMarchingPipeline(nn.Module):
    """
    Full pipeline combining P2VAE and FMT for PDE prediction.

    Handles:
    1. Encoding input fields to latent space
    2. Flow marching in latent space
    3. Decoding latent predictions back to field space
    """

    def __init__(self, vae, fmt):
        super().__init__()
        self.vae = vae
        self.fmt = fmt

    @torch.no_grad()
    def encode_frames(self, frames: list) -> list:
        """Encode a list of physical fields to latent space."""
        return [self.vae.get_latent(f, deterministic=True) for f in frames]

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to physical field."""
        return self.vae.decode(z)

    def training_step(
        self,
        frames: list,
        k_values: Optional[torch.Tensor] = None,
        t_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute conditional flow marching training loss.

        Args:
            frames: list of 4 physical field tensors (B, C, H, W)
            k_values: (B, 4) bridge parameters for each frame, or None for random
            t_values: (B, 4) time values for each frame, or None for random

        Returns:
            loss: scalar training loss
        """
        B = frames[0].shape[0]
        device = frames[0].device

        # Encode all frames to latent space
        with torch.no_grad():
            latents = [self.vae.get_latent(f, deterministic=True) for f in frames]

        # Sample k and t for each frame if not provided
        if k_values is None:
            k_values = torch.rand(B, 4, device=device)
        if t_values is None:
            t_values = torch.rand(B, 4, device=device)

        # Build noisy latent states for each frame
        noisy_latents = []
        for s in range(4):
            if s < 3:
                # Context frames: use x_s as x0, x_{s+1} as x1
                x0 = latents[s]
                x1 = latents[s + 1]
            else:
                # Target frame: use x_3 as x0, x_4 would be x1 (but we only have 4 frames)
                # For training, we use x_3 as the target
                x0 = latents[s - 1]
                x1 = latents[s]

            t_s = t_values[:, s]
            k_s = k_values[:, s]

            x_noisy, _, _ = sample_interpolation(x0, x1, t_s, k_s)
            noisy_latents.append(x_noisy)

        # Forward pass through FMT
        # The model predicts velocity for the last frame
        t_target = t_values[:, -1]
        velocity_pred, _ = self.fmt(noisy_latents, t_target)

        # Compute loss for the target frame
        x0_target = latents[-2]
        x1_target = latents[-1]
        t_s = t_values[:, -1]
        k_s = k_values[:, -1]
        x_noisy_target, _, _ = sample_interpolation(x0_target, x1_target, t_s, k_s)

        loss = flow_marching_loss(velocity_pred, x_noisy_target, x1_target, t_target)
        return loss
