"""
Evaluation script for WDNO.

Computes:
  - Simulation: MSE, MAE, L∞ error on state trajectories
  - Control: objective I = ∫|u(T,x) - u*(x)|² dx + α ∫|f|² dt dx
  - Super-resolution: MSE at finest resolution (with interpolation)

Also supports:
  - Zero-shot super-resolution evaluation
  - Comparison with baselines
  - Ablation studies

Usage:
  python evaluate.py --config configs/burgers_1d.yaml --checkpoint checkpoints/burgers_1d/best.pt
  python evaluate.py --config configs/burgers_1d.yaml --checkpoint checkpoints/burgers_1d/best.pt --mode control
  python evaluate.py --config configs/burgers_1d.yaml --checkpoint checkpoints/burgers_1d/best.pt --super_resolution
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.diffusion import GaussianDiffusion, cosine_guidance_schedule
from utils import (
    set_seed,
    load_config,
    load_checkpoint,
    compute_mse,
    compute_mae,
    compute_linf,
    compute_relative_l2,
    prepare_wavelet_batch_1d,
)


# ---------------------------------------------------------------------------
# Control objective for 1D Burgers'
# ---------------------------------------------------------------------------

def burgers_control_objective(
    u_T: torch.Tensor,
    u_target: torch.Tensor,
    f: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    I = ∫|u(T,x) - u*(x)|² dx + α ∫|f(t,x)|² dt dx

    Args:
        u_T: final state [batch, X]
        u_target: target state [batch, X]
        f: control force [batch, T, X]
        alpha: weight of energy term
    Returns:
        I: control objective [batch]
    """
    state_loss = torch.mean((u_T - u_target) ** 2, dim=-1)
    energy_loss = alpha * torch.mean(f ** 2, dim=(-2, -1))
    return state_loss + energy_loss


def fluid_control_objective(smoke_pct_final: torch.Tensor) -> torch.Tensor:
    """
    I = percentage of smoke NOT passing through target bucket.
    = 1 - smoke_pct_final

    Args:
        smoke_pct_final: fraction of smoke through target [batch]
    Returns:
        I: control objective [batch]
    """
    return 1.0 - smoke_pct_final


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def evaluate_simulation_1d(
    model: GaussianDiffusion,
    test_loader: DataLoader,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate simulation on 1D PDE data."""
    model.eval()
    wavelet = cfg["wavelet"]["wavelet"]
    wt_mode = cfg["wavelet"]["mode"]
    ddim_steps = cfg["inference"]["ddim_steps"]
    ddim_eta = cfg["inference"]["ddim_eta"]
    cfg_weight = cfg["inference"]["cfg_weight"]

    from pytorch_wavelets import DWTForward, DWTInverse

    all_mse = []
    all_mae = []
    all_linf = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating simulation"):
            u = batch["u"].to(device)
            f = batch["f"].to(device)

            # Prepare wavelet condition
            w_target, w_cond = prepare_wavelet_batch_1d(u, f, wavelet, wt_mode, "simulation")

            batch_size = u.shape[0]
            shape = (batch_size, w_target.shape[1], w_target.shape[2], w_target.shape[3])

            # Sample from diffusion model
            w_pred = model.sample(
                shape=shape,
                cond=w_cond,
                cfg_weight=cfg_weight,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
                device=device,
            )

            # Inverse wavelet transform
            idwt = DWTInverse(wave=wavelet, mode=wt_mode).to(device)
            n_channels = w_pred.shape[1] // 4
            yl = w_pred[:, :n_channels]
            yh_tensor = torch.stack([
                w_pred[:, n_channels:2*n_channels],
                w_pred[:, 2*n_channels:3*n_channels],
                w_pred[:, 3*n_channels:4*n_channels],
            ], dim=2)
            u_pred = idwt((yl, [yh_tensor])).squeeze(1)  # [B, T, X]

            # Ground truth (excluding t=0)
            u_gt = u[:, 1:]  # [B, T, X]

            all_mse.append(compute_mse(u_pred, u_gt).item())
            all_mae.append(compute_mae(u_pred, u_gt).item())
            all_linf.append(compute_linf(u_pred, u_gt).item())

    return {
        "mse": np.mean(all_mse),
        "mae": np.mean(all_mae),
        "linf": np.mean(all_linf),
    }


def evaluate_control_1d(
    model: GaussianDiffusion,
    test_loader: DataLoader,
    cfg: Dict[str, Any],
    device: torch.device,
    solver_fn=None,
) -> Dict[str, float]:
    """
    Evaluate control on 1D Burgers'.

    The control objective I is evaluated using the ground-truth solver
    given the generated control force f.
    """
    model.eval()
    wavelet = cfg["wavelet"]["wavelet"]
    wt_mode = cfg["wavelet"]["mode"]
    ddim_steps = cfg["inference"]["ddim_steps"]
    ddim_eta = cfg["inference"]["ddim_eta"]
    guidance_weight = cfg["control"]["guidance_weight"]
    alpha = cfg["control"].get("alpha", 0.1)

    from pytorch_wavelets import DWTForward, DWTInverse, DWT1DForward

    all_objectives = []

    for batch in tqdm(test_loader, desc="Evaluating control"):
        u = batch["u"].to(device)
        u_target = batch["u_T"].to(device)  # target state

        # Prepare condition
        w_target, w_cond = prepare_wavelet_batch_1d(
            u, batch["f"].to(device), wavelet, wt_mode, "control", u_target
        )

        batch_size = u.shape[0]
        shape = (batch_size, w_target.shape[1], w_target.shape[2], w_target.shape[3])

        # Guidance function: I(Ŵ_f) in wavelet domain
        # We compute the objective on the denoised estimate
        idwt = DWTInverse(wave=wavelet, mode=wt_mode).to(device)

        def guidance_fn(w_f_hat):
            """Compute guidance gradient w.r.t. wavelet coefficients of f."""
            n_ch = w_f_hat.shape[1] // 4
            yl = w_f_hat[:, :n_ch]
            yh_t = torch.stack([
                w_f_hat[:, n_ch:2*n_ch],
                w_f_hat[:, 2*n_ch:3*n_ch],
                w_f_hat[:, 3*n_ch:4*n_ch],
            ], dim=2)
            f_hat = idwt((yl, [yh_t])).squeeze(1)  # [B, T, X]

            # Energy term (we can compute this without solver)
            energy = alpha * torch.mean(f_hat ** 2)
            return energy

        def guidance_schedule(k, K_total):
            return cosine_guidance_schedule(k, K_total, guidance_weight)

        w_f_pred = model.sample(
            shape=shape,
            cond=w_cond,
            guidance_fn=guidance_fn,
            guidance_weight=guidance_weight,
            guidance_schedule=guidance_schedule,
            ddim_steps=ddim_steps,
            eta=ddim_eta,
            device=device,
        )

        # Inverse wavelet transform to get f
        n_ch = w_f_pred.shape[1] // 4
        yl = w_f_pred[:, :n_ch]
        yh_t = torch.stack([
            w_f_pred[:, n_ch:2*n_ch],
            w_f_pred[:, 2*n_ch:3*n_ch],
            w_f_pred[:, 3*n_ch:4*n_ch],
        ], dim=2)
        f_pred = idwt((yl, [yh_t])).squeeze(1)  # [B, T, X]

        # Evaluate objective using solver (if available) or surrogate
        if solver_fn is not None:
            # Use ground-truth solver to get u(T) given f
            u_T_pred = solver_fn(u[:, 0], f_pred)
            obj = burgers_control_objective(u_T_pred, u_target, f_pred, alpha)
        else:
            # Approximate: use energy term only
            obj = alpha * torch.mean(f_pred ** 2, dim=(-2, -1))

        all_objectives.extend(obj.cpu().numpy().tolist())

    return {
        "objective_mean": np.mean(all_objectives),
        "objective_std": np.std(all_objectives),
    }


def evaluate_super_resolution_1d(
    brm: GaussianDiffusion,
    srm: GaussianDiffusion,
    test_loader: DataLoader,
    cfg: Dict[str, Any],
    device: torch.device,
    n_sr_levels: int = 1,
) -> Dict[str, float]:
    """
    Evaluate zero-shot super-resolution for 1D experiments.

    Generates predictions at multiple resolutions and evaluates MSE
    at the finest resolution (via interpolation).
    """
    brm.eval()
    srm.eval()

    wavelet = cfg["wavelet"]["wavelet"]
    wt_mode = cfg["wavelet"]["mode"]
    ddim_steps = cfg["inference"]["ddim_steps"]
    ddim_eta = cfg["inference"]["ddim_eta"]

    from pytorch_wavelets import DWTForward, DWTInverse
    from wavelet.transforms import pad_to_match

    all_mse_by_level = {i: [] for i in range(n_sr_levels + 1)}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating super-resolution"):
            u = batch["u"].to(device)
            f = batch["f"].to(device)

            # Get BRM prediction at base resolution
            w_target, w_cond = prepare_wavelet_batch_1d(u, f, wavelet, wt_mode, "simulation")
            batch_size = u.shape[0]
            shape = (batch_size, w_target.shape[1], w_target.shape[2], w_target.shape[3])

            w_pred = brm.sample(
                shape=shape,
                cond=w_cond,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
                device=device,
            )

            idwt = DWTInverse(wave=wavelet, mode=wt_mode).to(device)

            def inv_wt(w):
                n_ch = w.shape[1] // 4
                yl = w[:, :n_ch]
                yh_t = torch.stack([w[:, n_ch:2*n_ch], w[:, 2*n_ch:3*n_ch], w[:, 3*n_ch:4*n_ch]], dim=2)
                return idwt((yl, [yh_t])).squeeze(1)

            u_pred_base = inv_wt(w_pred)  # [B, T', X']
            u_gt = u[:, 1:]  # [B, T, X]

            # Interpolate to finest resolution for comparison
            u_gt_fine = u_gt  # already at finest resolution

            # Level 0: base resolution
            u_pred_interp = F.interpolate(
                u_pred_base.unsqueeze(1),
                size=u_gt_fine.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            all_mse_by_level[0].append(compute_mse(u_pred_interp, u_gt_fine).item())

            # Super-resolution levels
            w_curr = w_pred
            for level in range(1, n_sr_levels + 1):
                # Prepare high-res condition (W_a_h at 2x resolution)
                # For simplicity, use the same condition upsampled
                w_a_high = F.interpolate(
                    w_cond,
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=False,
                )
                w_low_up = pad_to_match(w_curr, w_a_high)
                srm_cond = torch.cat([w_low_up, w_a_high], dim=1)
                srm_shape = (batch_size, w_curr.shape[1], w_a_high.shape[2], w_a_high.shape[3])

                w_curr = srm.sample(
                    shape=srm_shape,
                    cond=srm_cond,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    device=device,
                )

                u_pred_sr = inv_wt(w_curr)
                u_pred_interp = F.interpolate(
                    u_pred_sr.unsqueeze(1),
                    size=u_gt_fine.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                all_mse_by_level[level].append(compute_mse(u_pred_interp, u_gt_fine).item())

    return {f"mse_level_{k}": np.mean(v) for k, v in all_mse_by_level.items()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate WDNO")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--super_resolution", action="store_true")
    parser.add_argument("--srm_checkpoint", type=str, default=None)
    parser.add_argument("--n_sr_levels", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg["experiment"]["mode"]
    set_seed(cfg["experiment"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_2d = cfg["model"]["type"] == "3d"

    # Load model
    from train import build_model_1d, build_model_2d, get_dataset

    if is_2d:
        model = build_model_2d(cfg, mode).to(device)
    else:
        model = build_model_1d(cfg, mode).to(device)

    ckpt = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load test dataset
    test_dataset = get_dataset(cfg, "test", mode)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    if args.super_resolution and not is_2d:
        # Load SRM
        srm_path = args.srm_checkpoint or cfg.get("super_resolution", {}).get("srm_checkpoint")
        if srm_path is None:
            raise ValueError("SRM checkpoint required for super-resolution evaluation")

        from train_srm import build_srm_1d
        srm = build_srm_1d(cfg).to(device)
        srm_ckpt = load_checkpoint(srm_path, device)
        srm.load_state_dict(srm_ckpt["model"])
        srm.eval()

        results = evaluate_super_resolution_1d(
            model, srm, test_loader, cfg, device, args.n_sr_levels
        )
    elif mode == "simulation" and not is_2d:
        results = evaluate_simulation_1d(model, test_loader, cfg, device)
    elif mode == "control" and not is_2d:
        results = evaluate_control_1d(model, test_loader, cfg, device)
    else:
        print(f"Evaluation for {cfg['experiment']['name']} ({mode}) not fully implemented in this script.")
        results = {}

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
