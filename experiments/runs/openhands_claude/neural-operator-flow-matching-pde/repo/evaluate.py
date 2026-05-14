"""
Evaluation utilities:
  - L2 relative error (L2RE)
  - Variance-normalized RMSE (VRMSE)
  - Autoregressive rollout evaluation
  - Ensemble generation at different k3 values
  - Vorticity computation for visualization

Metrics follow the paper's definitions and match the evaluation in
The Well benchmark [34] and VICON [5].
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import DatasetConfig, get_fmt_config, get_p2vae_config
from data import PDEDataset, collate_traj
from model import FMT, P2VAE


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def l2_relative_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L2 relative error (L2RE).

    L2RE = ||pred - target||_2 / ||target||_2

    Args:
        pred, target: (B, C, H, W) or (B, ...) tensors

    Returns:
        l2re: (B,) per-sample errors
    """
    diff = (pred - target).reshape(pred.shape[0], -1)
    norm = target.reshape(target.shape[0], -1)
    return diff.norm(dim=-1) / (norm.norm(dim=-1) + 1e-8)


def vrmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Variance-normalized RMSE (VRMSE).

    VRMSE = RMSE(pred, target) / std(target)

    where std is computed over the spatial dimensions.

    Args:
        pred, target: (B, C, H, W)

    Returns:
        vrmse: (B,) per-sample errors
    """
    B = pred.shape[0]
    diff = (pred - target).reshape(B, -1)
    rmse = diff.pow(2).mean(dim=-1).sqrt()

    target_flat = target.reshape(B, -1)
    std = target_flat.std(dim=-1) + 1e-8
    return rmse / std


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Compute both L2RE and VRMSE, return mean over batch."""
    l2re = l2_relative_error(pred, target).mean().item()
    vrmse_val = vrmse(pred, target).mean().item()
    return {"l2re": l2re, "vrmse": vrmse_val}


# ---------------------------------------------------------------------------
# Vorticity computation
# ---------------------------------------------------------------------------

def compute_vorticity(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute vorticity ω = ∂v/∂x - ∂u/∂y using finite differences.

    Args:
        u, v: (B, H, W) velocity components

    Returns:
        omega: (B, H, W) vorticity field
    """
    # Central differences with periodic boundary
    dvdx = torch.roll(v, -1, dims=-1) - torch.roll(v, 1, dims=-1)
    dudy = torch.roll(u, -1, dims=-2) - torch.roll(u, 1, dims=-2)
    return (dvdx - dudy) / 2.0


# ---------------------------------------------------------------------------
# Autoregressive rollout evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_rollout(
    vae: P2VAE,
    fmt: FMT,
    dataloader: DataLoader,
    device: torch.device,
    n_rollout_steps: int = 14,
    k_val: float = 1.0,
    n_euler_steps: int = 100,
    max_batches: Optional[int] = None,
) -> Dict[str, List[float]]:
    """Evaluate autoregressive rollout over multiple steps.

    Returns per-step L2RE averaged over the dataset.

    Args:
        vae:             frozen P2VAE
        fmt:             trained FMT
        dataloader:      yields (traj_len, C, H, W) trajectories
        device:          compute device
        n_rollout_steps: number of future steps to predict
        k_val:           bridge parameter (1=deterministic, <1=stochastic)
        n_euler_steps:   Euler ODE steps per prediction
        max_batches:     limit evaluation batches (None = full dataset)

    Returns:
        metrics: dict mapping step index to list of L2RE values
    """
    vae.eval()
    fmt.eval()

    step_errors: Dict[int, List[float]] = {s: [] for s in range(n_rollout_steps)}

    for batch_idx, frames in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # frames: list of traj_len tensors, each (B, C, H, W)
        # Use first frame as context, rest as ground truth
        x_context = [frames[0].to(device)]
        x_gt = [f.to(device) for f in frames[1:]]

        # Encode context
        y_context = [vae.encode_deterministic(x) for x in x_context]

        # Rollout
        y_preds = fmt.rollout(
            y_context=y_context,
            n_future=n_rollout_steps,
            k_val=k_val,
            n_euler_steps=n_euler_steps,
        )

        # Decode predictions and compute errors
        for s, y_pred in enumerate(y_preds):
            x_pred = vae.decode(y_pred)

            if s < len(x_gt):
                x_true = x_gt[s]
            else:
                # No ground truth for this step
                continue

            l2re = l2_relative_error(x_pred, x_true).mean().item()
            step_errors[s].append(l2re)

    return step_errors


@torch.no_grad()
def evaluate_reconstruction(
    vae: P2VAE,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate P2VAE reconstruction quality."""
    vae.eval()
    all_l2re = []
    all_vrmse = []

    for batch_idx, frames in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = frames[0].to(device)
        x_hat = vae.decode(vae.encode_deterministic(x))

        all_l2re.append(l2_relative_error(x_hat, x).mean().item())
        all_vrmse.append(vrmse(x_hat, x).mean().item())

    return {
        "l2re": float(np.mean(all_l2re)),
        "vrmse": float(np.mean(all_vrmse)),
    }


# ---------------------------------------------------------------------------
# Ensemble generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_ensemble(
    vae: P2VAE,
    fmt: FMT,
    x_context: List[torch.Tensor],
    device: torch.device,
    ensemble_size: int = 32,
    k3_values: Optional[List[float]] = None,
    n_euler_steps: int = 100,
) -> Dict[float, torch.Tensor]:
    """Generate an ensemble of next-state predictions at different k3 values.

    Args:
        vae:           frozen P2VAE
        fmt:           trained FMT
        x_context:     list of 3 clean context frames, each (1, C, H, W)
        device:        compute device
        ensemble_size: number of samples per k3 value
        k3_values:     list of bridge parameters to sweep (default: [0, 0.3, 0.6, 0.9])
        n_euler_steps: Euler ODE steps

    Returns:
        ensembles: dict mapping k3 → (ensemble_size, C, H, W) decoded predictions
    """
    if k3_values is None:
        k3_values = [0.0, 0.3, 0.6, 0.9]

    vae.eval()
    fmt.eval()

    # Encode context frames
    y_context = [vae.encode_deterministic(x.to(device)) for x in x_context]

    # Pad to 4 frames
    while len(y_context) < fmt.cfg.n_frames:
        y_context.insert(0, y_context[0])

    # Build GRU state from clean context
    h = fmt.df_gru.init_hidden(1, device)
    for y in y_context:
        h = fmt.df_gru(h, y)

    ensembles = {}
    for k3 in k3_values:
        preds = []
        for _ in range(ensemble_size):
            window = y_context[-fmt.cfg.n_frames:]

            # Frames 0-2: clean; frame 3: noisy according to k3
            y_init_list = [window[s].clone() for s in range(fmt.cfg.n_frames - 1)]
            y_prev = window[-1]
            if k3 < 1.0:
                z = torch.randn_like(y_prev)
                y3_init = k3 * y_prev + (1.0 - k3) * z
            else:
                y3_init = y_prev.clone()
            y_init_list.append(y3_init)

            h_list = [h] * fmt.cfg.n_frames
            y_final_list = fmt.euler_sample(
                y_init_list,
                k_list=[1.0] * (fmt.cfg.n_frames - 1) + [k3],
                h_list=h_list,
                n_steps=n_euler_steps,
            )
            y_next = y_final_list[-1]
            x_next = vae.decode(y_next)
            preds.append(x_next)

        ensembles[k3] = torch.cat(preds, dim=0)  # (ensemble_size, C, H, W)

    return ensembles


def compute_ensemble_variance(ensembles: Dict[float, torch.Tensor]) -> Dict[float, float]:
    """Compute average batch-wise variance of ensemble predictions.

    Matches Fig. 3 in the paper: variance as a function of k3.
    """
    variances = {}
    for k3, samples in ensembles.items():
        # samples: (ensemble_size, C, H, W)
        var = samples.var(dim=0).mean().item()
        variances[k3] = var
    return variances


# ---------------------------------------------------------------------------
# Full benchmark evaluation (Table 3 style)
# ---------------------------------------------------------------------------

def run_benchmark(
    vae_ckpt: str,
    fmt_ckpt: str,
    data_dir: str,
    dataset_names: List[str],
    device: torch.device,
    n_rollout_steps: int = 14,
    batch_size: int = 16,
    max_batches: int = 100,
) -> None:
    """Run the long-term rollout benchmark across multiple datasets."""
    # Load models
    vae_state = torch.load(vae_ckpt, map_location=device)
    vae = P2VAE(vae_state["cfg"]).to(device)
    vae.load_state_dict(vae_state["model_state"])
    vae.eval()

    fmt_state = torch.load(fmt_ckpt, map_location=device)
    fmt = FMT(fmt_state["cfg"]).to(device)
    fmt.load_state_dict(fmt_state["model_state"])
    fmt.eval()

    print(f"\n{'Dataset':<20} {'Step 1':>8} {'Step 5':>8} {'Step 10':>8} {'Last':>8} {'Avg':>8}")
    print("-" * 65)

    for name in dataset_names:
        path = os.path.join(data_dir, f"{name}.h5")
        if not os.path.exists(path):
            print(f"{name:<20} (file not found)")
            continue

        ds = PDEDataset(path, traj_len=4, split="test")
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_traj, num_workers=4)

        step_errors = evaluate_rollout(
            vae, fmt, loader, device,
            n_rollout_steps=n_rollout_steps,
            max_batches=max_batches,
        )

        # Aggregate
        means = {s: float(np.mean(v)) for s, v in step_errors.items() if v}
        s1 = means.get(0, float("nan"))
        s5 = means.get(4, float("nan"))
        s10 = means.get(9, float("nan"))
        last = means.get(max(means.keys()), float("nan"))
        avg = float(np.mean(list(means.values())))

        print(f"{name:<20} {s1:>8.4f} {s5:>8.4f} {s10:>8.4f} {last:>8.4f} {avg:>8.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FMT on PDE benchmarks")
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--fmt_ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="/data/pde")
    parser.add_argument("--datasets", nargs="+",
                        default=["pa_ns", "pb_cns_low", "pb_cns_high"])
    parser.add_argument("--n_rollout_steps", type=int, default=14)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    run_benchmark(
        vae_ckpt=args.vae_ckpt,
        fmt_ckpt=args.fmt_ckpt,
        data_dir=args.data_dir,
        dataset_names=args.datasets,
        device=device,
        n_rollout_steps=args.n_rollout_steps,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )
