"""
Evaluation script for FMT model.

Evaluates:
1. P2VAE reconstruction quality (L2RE, VRMSE)
2. FMT prediction quality at different rollout steps (1, 5, 10, last)
3. Long-term rollout stability

Evaluation settings from the paper:
- Euler ODE sampler with N=100 steps, dt=0.01
- Deterministic prediction: k=(1,1,1,1)
- Generation: k=(1,1,1,k3) with k3 < 1

Inference procedure:
1. For each autoregressive step:
   a. Update diffusion forcing state h using context frames
   b. Run Euler ODE integration (100 steps) to predict next state
   c. Pass h to next autoregressive step
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.p2vae import P2VAE, P2VAE_16M, P2VAE_87M
from models.fmt import FlowMarchingTransformer, FMTSmall, FMTBase, FMTLarge
from evaluation.metrics import l2_relative_error, vrmse, PDEMetrics, compute_vorticity
from data.pde_dataset import SinglePDEDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_vae_reconstruction(
    vae: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate P2VAE reconstruction quality.

    Args:
        vae: P2VAE model
        dataloader: data loader
        device: computation device
        max_batches: maximum number of batches to evaluate

    Returns:
        dict with 'l2re' and 'vrmse' metrics
    """
    vae.eval()
    metrics = PDEMetrics()

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        # batch: (B, T, C, H, W) or (B, C, H, W)
        if batch.dim() == 5:
            B, T, C, H, W = batch.shape
            x = batch[:, 0].to(device)  # Use first frame
        else:
            x = batch.to(device)

        # Reconstruct
        x_hat, _ = vae(x)

        metrics.update(x_hat, x)

    return metrics.compute()


@torch.no_grad()
def predict_next_state(
    fmt: torch.nn.Module,
    context_latents: List[torch.Tensor],
    h_state: Optional[torch.Tensor],
    device: torch.device,
    num_ode_steps: int = 100,
    k_target: float = 1.0,
) -> tuple:
    """
    Predict the next state using Euler ODE integration.

    The diffusion forcing state h is updated once using the context frames,
    then the ODE integration runs with fixed h.

    Args:
        fmt: FlowMarchingTransformer
        context_latents: list of 3 context latent tensors (B, C, H, W)
        h_state: (B, embed_dim) previous diffusion forcing state
        device: computation device
        num_ode_steps: number of Euler ODE steps
        k_target: bridge parameter for target frame

    Returns:
        pred_latent: (B, C, H, W) predicted next latent state
        h_new: (B, embed_dim) updated diffusion forcing state
    """
    B = context_latents[0].shape[0]
    dt = 1.0 / num_ode_steps

    # Initialize target frame
    if k_target < 1.0:
        # Start from noisy initialization
        noise = torch.randn_like(context_latents[-1])
        x_t = k_target * context_latents[-1] + (1 - k_target) * noise
    else:
        # Deterministic: start from last context frame
        x_t = context_latents[-1].clone()

    # First, update h_state using context frames (before ODE integration)
    # This is done by running the model once with t=0 to get the updated h
    t_zero = torch.zeros(B, device=device)
    input_frames = list(context_latents) + [x_t]
    _, h_new = fmt(input_frames, t_zero, h_state)

    # Euler ODE integration with fixed h_new
    for ode_step in range(num_ode_steps):
        t_tensor = torch.full((B,), ode_step * dt, device=device)
        # Use h_new (fixed) for all ODE steps
        velocity, _ = fmt(input_frames[:-1] + [x_t], t_tensor, h_state)
        x_t = x_t + dt * velocity

    return x_t, h_new


@torch.no_grad()
def evaluate_fmt_prediction(
    vae: torch.nn.Module,
    fmt: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_rollout_steps: int = 14,
    num_ode_steps: int = 100,
    k_values: tuple = (1.0, 1.0, 1.0, 1.0),
    max_trajectories: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate FMT prediction quality with long-term rollout.

    Args:
        vae: P2VAE model (frozen)
        fmt: FMT model
        dataloader: data loader (trajectories of length >= num_rollout_steps + 3)
        device: computation device
        num_rollout_steps: number of autoregressive prediction steps
        num_ode_steps: number of Euler ODE steps per prediction
        k_values: bridge parameters (k0, k1, k2, k3) for each context frame
        max_trajectories: maximum number of trajectories to evaluate

    Returns:
        dict mapping step -> {'l2re': float, 'vrmse': float}
    """
    vae.eval()
    fmt.eval()

    # Track metrics at different steps
    step_metrics = {
        1: PDEMetrics(),
        5: PDEMetrics(),
        10: PDEMetrics(),
        "last": PDEMetrics(),
        "average": PDEMetrics(),
    }

    traj_count = 0

    for batch in dataloader:
        if max_trajectories is not None and traj_count >= max_trajectories:
            break

        # batch: (B, T, C, H, W)
        batch = batch.to(device)
        B, T, C, H, W = batch.shape

        if T < num_rollout_steps + 3:
            logger.warning(f"Trajectory too short: {T} < {num_rollout_steps + 3}")
            continue

        # Encode all frames
        latents = []
        for t_idx in range(T):
            z = vae.get_latent(batch[:, t_idx], deterministic=True)
            latents.append(z)

        # Initialize with first 3 frames as context
        context_latents = list(latents[:3])
        h_state = None

        # Autoregressive rollout
        all_preds = []

        for step in range(num_rollout_steps):
            # Get 3 context frames
            if step == 0:
                ctx = context_latents
            else:
                # Use predicted frames as context (sliding window)
                all_available = context_latents + all_preds
                ctx = all_available[-3:]

            # Predict next state using Euler ODE
            pred_latent, h_state = predict_next_state(
                fmt, ctx, h_state, device,
                num_ode_steps=num_ode_steps,
                k_target=k_values[-1],
            )

            all_preds.append(pred_latent)

            # Decode prediction
            pred_field = vae.decode(pred_latent)
            target_field = batch[:, step + 3]  # Ground truth at step+3

            # Compute metrics at specific steps
            step_num = step + 1
            if step_num in step_metrics:
                step_metrics[step_num].update(pred_field, target_field)

            # Always update average
            step_metrics["average"].update(pred_field, target_field)

        # Update "last" step metrics
        if all_preds:
            last_pred = vae.decode(all_preds[-1])
            last_target = batch[:, num_rollout_steps + 2]
            step_metrics["last"].update(last_pred, last_target)

        traj_count += B

    # Compute final metrics
    results = {}
    for step_key, metric in step_metrics.items():
        results[step_key] = metric.compute()

    return results


@torch.no_grad()
def evaluate_ensemble_generation(
    vae: torch.nn.Module,
    fmt: torch.nn.Module,
    context_frames: torch.Tensor,
    device: torch.device,
    k3_values: List[float] = [0.0, 0.3, 0.6, 0.9],
    ensemble_size: int = 32,
    num_ode_steps: int = 100,
) -> Dict[float, torch.Tensor]:
    """
    Generate ensemble of next states at different noise levels k3.

    From the paper: "By tuning bridge parameter k3 during the generation,
    we can effectively generate an ensemble of possible next state given
    a noisy initialization k3*x3 + (1-k3)*z and concluded PDE condition h3
    from clean past frames (x0, x1, x2)."

    The variance of the predicted ensemble is a decreasing function of k3.

    Args:
        vae: P2VAE model
        fmt: FMT model
        context_frames: (1, 3, C, H, W) or (3, C, H, W) context frames
        device: computation device
        k3_values: list of k3 values to test [0, 0.3, 0.6, 0.9]
        ensemble_size: number of samples per k3 value (32 in paper)
        num_ode_steps: number of Euler ODE steps

    Returns:
        dict mapping k3 -> (ensemble_size, C, H, W) generated fields
    """
    vae.eval()
    fmt.eval()

    if context_frames.dim() == 4:
        context_frames = context_frames.unsqueeze(0)

    # Encode context frames
    context_latents = []
    for i in range(3):
        z = vae.get_latent(context_frames[:, i], deterministic=True)
        context_latents.append(z.expand(ensemble_size, -1, -1, -1))

    results = {}

    for k3 in k3_values:
        # Initialize target frame with noise level k3
        noise = torch.randn_like(context_latents[-1])

        if k3 >= 1.0:
            x_t = context_latents[-1].clone()
        else:
            x_t = k3 * context_latents[-1] + (1 - k3) * noise

        # Get conditioning state from clean context frames
        t_zero = torch.zeros(ensemble_size, device=device)
        input_frames = context_latents + [x_t]
        _, h_state = fmt(input_frames, t_zero, None)

        # Euler ODE integration
        dt = 1.0 / num_ode_steps

        for ode_step in range(num_ode_steps):
            t_tensor = torch.full((ensemble_size,), ode_step * dt, device=device)
            velocity, _ = fmt(context_latents + [x_t], t_tensor, None)
            x_t = x_t + dt * velocity

        # Decode
        generated = vae.decode(x_t)
        results[k3] = generated

    return results


def evaluate_all(args):
    """Run full evaluation pipeline."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load P2VAE
    if args.vae_size == "16M":
        vae = P2VAE_16M(in_channels=3, latent_channels=16)
    else:
        vae = P2VAE_87M(in_channels=3, latent_channels=16)

    if args.vae_checkpoint:
        checkpoint = torch.load(args.vae_checkpoint, map_location="cpu")
        vae.load_state_dict(checkpoint["model_state_dict"])
    vae = vae.to(device)
    vae.eval()

    # Load FMT
    if args.fmt_size == "S":
        fmt = FMTSmall(latent_channels=16, latent_size=16)
    elif args.fmt_size == "B":
        fmt = FMTBase(latent_channels=16, latent_size=16)
    else:
        fmt = FMTLarge(latent_channels=16, latent_size=16)

    if args.fmt_checkpoint:
        checkpoint = torch.load(args.fmt_checkpoint, map_location="cpu")
        fmt.load_state_dict(checkpoint["model_state_dict"])
    fmt = fmt.to(device)
    fmt.eval()

    # Evaluate on each dataset
    for dataset_name in args.datasets:
        logger.info(f"\nEvaluating on {dataset_name}...")

        try:
            dataset = SinglePDEDataset(
                data_path=f"{args.data_dir}/{dataset_name.lower().replace('-', '_')}",
                split="test",
                trajectory_length=args.rollout_steps + 3,
            )
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

            # VAE reconstruction
            vae_metrics = evaluate_vae_reconstruction(vae, loader, device, max_batches=50)
            logger.info(f"  VAE Reconstruction - L2RE: {vae_metrics['l2re']:.4f}, VRMSE: {vae_metrics['vrmse']:.4f}")

            # FMT prediction
            pred_metrics = evaluate_fmt_prediction(
                vae, fmt, loader, device,
                num_rollout_steps=args.rollout_steps,
                num_ode_steps=args.ode_steps,
                max_trajectories=args.max_trajectories,
            )

            for step_key, metrics in pred_metrics.items():
                logger.info(f"  Step {step_key} - L2RE: {metrics['l2re']:.4f}, VRMSE: {metrics['vrmse']:.4f}")

        except Exception as e:
            logger.error(f"  Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FMT model")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--vae_checkpoint", type=str, default=None)
    parser.add_argument("--fmt_checkpoint", type=str, default=None)
    parser.add_argument("--vae_size", type=str, default="16M", choices=["16M", "87M"])
    parser.add_argument("--fmt_size", type=str, default="B", choices=["S", "B", "L"])
    parser.add_argument("--datasets", nargs="+", default=["PA-NS", "PB-CNSL", "PB-CNSH"])
    parser.add_argument("--rollout_steps", type=int, default=14)
    parser.add_argument("--ode_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_trajectories", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_all(args)
