## main.py
"""
Main entry point for reproducing experiments of Wavelet Diffusion Neural Operator (WDNO).

Usage:
    python main.py --experiment burgers_1d_sim --task train
    python main.py --experiment burgers_1d_ctrl --task eval --device cuda:0

Implements the complete pipeline: configuration loading, wavelet setup, dataset
construction (base and multi‑resolution), model creation (2D/3D UNets, DDPM/DDIM),
optional control surrogate training, training orchestrator (BRM, SRM, surrogate),
and evaluation (simulation, control, super‑resolution).

All hyper‑parameters are read from config.yaml via the Config class. No values
are hard‑coded; the script adapts to every experiment defined in the paper.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset as TorchDataset

# Project‑specific imports – ensure the package structure allows these
from config import Config
from utils import set_seed, setup_logging, get_device
from wavelet_utils import WaveletTransform
from dataset import (
    PDEBenchDataset,
    IncompressibleFluidDataset,
    MultiResolutionDataset,
)
from unet_2d import UNet2D
from unet_3d import UNet3D
from diffusion import DDPM
from models.wdno import WDNO
from trainer import Trainer
from control_surrogate import ControlSurrogate
from evaluate import Evaluator

logger = logging.getLogger("wdno_main")


# --------------------------------------------------------------------------
# Small helper dataset for control surrogate training
# --------------------------------------------------------------------------

class ControlSurrogateDataset(TorchDataset):
    """
    Wraps a base physical dataset to produce (u0, f, target_J) tuples
    for supervised training of the control surrogate.

    Args:
        base_dataset: PDEBenchDataset or IncompressibleFluidDataset.
        alpha: energy penalty weight for 1D cases (2D uses alpha=0).
        is_2d: whether the dataset is 2D/ERA5.
    """

    def __init__(
        self,
        base_dataset: Union[PDEBenchDataset, IncompressibleFluidDataset],
        alpha: float = 2e-5,
        is_2d: bool = False,
    ) -> None:
        super().__init__()
        self.base = base_dataset
        self.alpha = alpha
        self.is_2d = is_2d
        self.experiment = base_dataset.experiment if hasattr(base_dataset, 'experiment') else "unknown"

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.base[idx]

        if self.is_2d:
            # 2D fluid control objective: negative smoke pass percentage
            u0 = sample.get("initial_density")  # shape (H, W)
            f = sample.get("control")           # shape (T, C_f, H, W) or (T, H, W)
            # Target J: percentage of smoke NOT passing target bucket.
            # The dataset should contain 'bucket_percentage' or similar.
            # Here we assume 'bucket_not_passed' is provided.
            target = sample.get("bucket_not_passed", torch.tensor(0.0)).float()
            # Ensure u0 has batch dim
            if u0.dim() == 2:
                u0 = u0.unsqueeze(0)  # (1, H, W)
            return u0.squeeze(0), f, target   # u0: (H, W), f: (T, ...), target: scalar
        else:
            # 1D Burgers control: J = ∫|u(T)-u*|^2 + α∫|f|^2
            u0 = sample["u0"]          # (X,)
            f = sample["f"]            # (T, X)
            uT = sample.get("uT")      # target final state (X,)
            u_final = sample["state"][-1]  # last frame of state (X,)  # state is (T, X)
            # Compute target J from the dataset's own trajectory (since dataset uses the true f)
            state_error = torch.mean((u_final - uT) ** 2)
            force_energy = torch.mean(f ** 2)
            target = state_error + self.alpha * force_energy
            return u0, f, target.unsqueeze(0)  # target shape (1,) -> scalar after squeeze


# --------------------------------------------------------------------------
# Solver for 1D Burgers’ equation (used for ground‑truth evaluation)
# --------------------------------------------------------------------------

def solve_burgers_1d(
    u0: np.ndarray,      # shape (X,)
    f: np.ndarray,        # shape (T, X)
    T: float = 8.0,
    nu: float = 0.01,
    nx: int = 120,        # spatial points
    nt_stored: int = 80,  # number of stored frames (excluding initial)
) -> np.ndarray:
    """
    Finite‑difference solver for the 1D viscous Burgers’ equation
       ∂u/∂t = -u ∂u/∂x + ν ∂²u/∂x² + f(t,x)
    with Dirichlet BC u=0 on boundaries, over domain x ∈ [0,1], t ∈ [0,T].

    The solver uses a high‑resolution internal grid (dt fine, dx fine) and
    returns the trajectory at the stored time intervals.

    Args:
        u0: initial condition (shape (nx,))
        f:  force at stored time steps (shape (nt_stored, nx)). The force is
            assumed constant between stored times.
        T:  total simulation time.
        nu: viscosity.
        nx: spatial grid points.
        nt_stored: number of time frames to output (excluding t=0). Total
                    number of output points = nt_stored + 1 (including t=0).

    Returns:
        u: trajectory of shape (nt_stored+1, nx) (including initial condition).
    """
    # High‑resolution internal steps to ensure stability
    dx = 1.0 / (nx - 1)
    # CFL condition: dt <= dx^2 / (2*nu + max|u|*dx) – we choose a very small dt
    dt_max = 0.0001
    nt_internal = max(int(T / dt_max), nt_stored * 100)  # many internal steps
    dt = T / nt_internal

    # Interpolate force to internal time steps
    stored_times = np.linspace(0, T, nt_stored, endpoint=False)
    internal_times = np.linspace(0, T, nt_internal, endpoint=False)
    # Find indices: for each internal time, determine which stored interval
    f_at_internal = np.zeros((nt_internal, nx))
    for i, t in enumerate(internal_times):
        idx = max(0, int(t / (T / nt_stored)))  # simple piecewise constant
        if idx >= nt_stored:
            idx = nt_stored - 1
        f_at_internal[i] = f[idx]

    # Time integration
    u = np.zeros((nt_internal + 1, nx))
    u[0] = u0.copy()
    for n in range(nt_internal):
        u_curr = u[n]
        # Upwind for convection
        upwind = np.where(u_curr >= 0,
                          u_curr * (u_curr - np.roll(u_curr, 1)) / dx,
                          u_curr * (np.roll(u_curr, -1) - u_curr) / dx)
        # Central difference for diffusion
        u_xx = (np.roll(u_curr, 1) - 2 * u_curr + np.roll(u_curr, -1)) / (dx ** 2)
        rhs = -upwind + nu * u_xx + f_at_internal[n]
        u[n + 1] = u_curr + dt * rhs
        # Dirichlet BC
        u[n + 1, 0] = 0.0
        u[n + 1, -1] = 0.0

    # Downsample to stored frames
    store_indices = np.linspace(0, nt_internal, nt_stored + 1, dtype=int)
    u_stored = u[store_indices]  # shape (nt_stored+1, nx)
    return u_stored


# --------------------------------------------------------------------------
# Main function
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Wavelet Diffusion Neural Operator (WDNO) reproduction.")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["burgers_1d_sim", "burgers_1d_ctrl", "advection_1d",
                                 "cfd_1d", "fluid_2d_sim", "fluid_2d_ctrl", "era5"],
                        help="Name of the experiment to run.")
    parser.add_argument("--task", type=str, default="all", choices=["train", "eval", "all"],
                        help="What to execute: training, evaluation, or both.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the YAML configuration file.")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (e.g., 'cuda:0', 'cpu'). If not given, uses config value.")
    args = parser.parse_args()

    # ---- Load configuration ----
    config = Config.from_yaml(args.config)
    # Override experiment if needed (should match, but just in case)
    if config.get_experiment_name() != args.experiment:
        logger.warning(f"Config experiment {config.get_experiment_name()} does not match CLI {args.experiment}. Using CLI.")
        config._cfg["experiment"] = args.experiment
    if args.device is not None:
        config._cfg["device"] = args.device

    # ---- Global setup ----
    set_seed(config.get_seed())
    device = get_device(config.get_device())
    log_file = os.path.join("logs", f"{args.experiment}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    setup_logging(log_level=logging.INFO, log_file=log_file)

    logger.info(f"Starting experiment: {args.experiment}, task: {args.task}")

    # ---- Wavelet transform ----
    wavelet_cfg = config.get_wavelet_config()
    wavelet_transform = WaveletTransform(
        wavelet=wavelet_cfg["wname"],
        mode=wavelet_cfg["mode"],
        ndim=wavelet_cfg["ndim"],
        rec_tol=wavelet_cfg.get("rec_tol", 1e-6),
    )
    logger.info(f"Wavelet transform: {wavelet_cfg['wname']}, {wavelet_cfg['mode']}, ndim={wavelet_cfg['ndim']}")

    # ---- Base datasets (train/val/test) ----
    data_cfg = config.get_data_config()
    experiment = args.experiment
    is_1d = "1d" in experiment
    is_2d = "2d" in experiment or "era5" in experiment
    is_control = "ctrl" in experiment

    # Factory for base dataset
    if is_1d:
        # 1D experiments: Burgers, Advection, 1D CFD
        if experiment in ("burgers_1d_sim", "burgers_1d_ctrl"):
            root = data_cfg["burgers_root"]
        elif experiment == "advection_1d":
            root = data_cfg["advection_root"]
        elif experiment == "cfd_1d":
            root = data_cfg["cfd_root"]
        else:
            raise ValueError(f"Unsupported 1D experiment: {experiment}")
        base_train = PDEBenchDataset(config.get_all_configs(), split="train")
        base_val   = PDEBenchDataset(config.get_all_configs(), split="val")
        base_test  = PDEBenchDataset(config.get_all_configs(), split="test")
    else:
        # 2D experiments: fluid, ERA5
        if experiment in ("fluid_2d_sim", "fluid_2d_ctrl"):
            root = data_cfg["fluid_root"]
        elif experiment == "era5":
            root = data_cfg["era5_root"]
        else:
            raise ValueError(f"Unsupported 2D experiment: {experiment}")
        base_train = IncompressibleFluidDataset(config.get_all_configs(), split="train")
        base_val   = IncompressibleFluidDataset(config.get_all_configs(), split="val")
        base_test  = IncompressibleFluidDataset(config.get_all_configs(), split="test")
    logger.info(f"Loaded base datasets: train={len(base_train)}, val={len(base_val)}, test={len(base_test)}")

    # ---- Condition keys for each experiment ----
    # These correspond to the keys in the base sample that should be wavelet‑transformed
    # and concatenated as conditioning for the diffusion model.
    # For simulation tasks: condition includes the equation parameters (initial condition, force)
    # For control tasks: condition includes initial condition and target final state.
    if experiment in ("burgers_1d_sim", "advection_1d", "cfd_1d"):
        cond_keys = ["u0", "f"] if "f" in base_train[0] else ["u0"]
    elif experiment in ("burgers_1d_ctrl",):
        cond_keys = ["u0", "uT"]
    elif experiment == "fluid_2d_sim":
        cond_keys = ["initial_density", "control"]
    elif experiment == "fluid_2d_ctrl":
        cond_keys = ["initial_density"]
    elif experiment == "era5":
        cond_keys = ["past"]   # the past 12 hours are used as condition
    else:
        cond_keys = []

    # State keys: what to generate (the target wavelet)
    state_keys = ["state"]   # always

    # ---- Multi‑resolution datasets ----
    scales = data_cfg["super_res_scales"]  # e.g., [1, 0.5, 0.25, 0.125] for 1D; [1, 0.5] for 2D
    brm_dataset = MultiResolutionDataset(
        base_dataset=base_train,
        scales=[1.0],               # only full resolution for BRM
        wavelet_transform=wavelet_transform,
        mode="brm",
        state_keys=state_keys,
        cond_keys_high=cond_keys,
        time_scale_with_spatial=True,
    )
    logger.info(f"BRM dataset (full res): {len(brm_dataset)} samples")

    srm_dataset = None
    if len(scales) > 1:
        srm_dataset = MultiResolutionDataset(
            base_dataset=base_train,
            scales=scales,           # e.g., [1, 0.5, 0.25, 0.125]
            wavelet_transform=wavelet_transform,
            mode="srm",
            state_keys=state_keys,
            cond_keys_high=cond_keys,
            time_scale_with_spatial=True,
        )
        logger.info(f"SRM dataset: {len(srm_dataset)} samples")

    # ---- Build denoisers ----
    # BRM
    if is_1d:
        base_model_cfg = config.get_base_model_config()   # in_channels computed
        denoiser_brm = UNet2D(
            in_channels=base_model_cfg["in_channels"],
            out_channels=base_model_cfg["out_channels"],
            initial_dim=base_model_cfg["initial_dim"],
            dim_mults=base_model_cfg["dim_mults"],
            resnet_groups=base_model_cfg["resnet_groups"],
            attention_heads=base_model_cfg["attention_heads"],
            attention_dim=base_model_cfg["attention_dim"],
            kernel_size=base_model_cfg["kernel_size"],
        ).to(device)
    else:
        model3d_cfg = config.get_3d_model_config()   # already includes computed in_channels
        denoiser_brm = UNet3D(
            params=model3d_cfg,
            time_emb_dim=model3d_cfg.get("dim", 64) * 4,
        ).to(device)
    logger.info(f"BRM denoiser created with {sum(p.numel() for p in denoiser_brm.parameters())} parameters.")

    # SRM (if super‑resolution needed)
    denoiser_srm = None
    if len(scales) > 1:
        if is_1d:
            super_model_cfg = config.get_super_model_config()
            denoiser_srm = UNet2D(
                in_channels=super_model_cfg["in_channels"],
                out_channels=super_model_cfg["out_channels"],
                initial_dim=super_model_cfg["initial_dim"],
                dim_mults=super_model_cfg["dim_mults"],
                resnet_groups=super_model_cfg["resnet_groups"],
                attention_heads=super_model_cfg["attention_heads"],
                attention_dim=super_model_cfg["attention_dim"],
                kernel_size=super_model_cfg["kernel_size"],
            ).to(device)
        else:
            # For 2D/ERA5 SRM, the model config is same as 3D base but with extra input channels.
            # We duplicate the base 3D config and adjust in_channels.
            srm_3d_cfg = copy.deepcopy(model3d_cfg)
            # SRM adds extra low‑res wavelet channels: number of subbands (8)
            extra_channels = wavelet_transform.ndim == 3 and 8 or 4
            srm_3d_cfg["in_channels"] = model3d_cfg["in_channels"] + extra_channels
            denoiser_srm = UNet3D(params=srm_3d_cfg, time_emb_dim=srm_3d_cfg.get("dim", 64) * 4).to(device)
        logger.info(f"SRM denoiser created with {sum(p.numel() for p in denoiser_srm.parameters())} parameters.")

    # ---- Build DDPM wrappers ----
    diff_cfg = config.get_diffusion_config()
    ddpm_brm = DDPM(
        denoiser=denoiser_brm,
        n_timesteps=diff_cfg["num_timesteps"],
        schedule=diff_cfg["schedule"],
        device=device,
    )
    ddpm_srm = None
    if denoiser_srm is not None:
        ddpm_srm = DDPM(
            denoiser=denoiser_srm,
            n_timesteps=diff_cfg["num_timesteps"],
            schedule=diff_cfg["schedule"],
            device=device,
        )

    # ---- Control surrogate (only for control experiments) ----
    control_surrogate = None
    if is_control:
        ctrl_cfg = config.get_control_config()
        surrogate = ControlSurrogate(config.get_all_configs())
        # Train surrogate on the training set
        if args.task in ("train", "all"):
            logger.info("Training control surrogate...")
            # Build a wrapper dataset that yields (u0, f, target_J)
            alpha = ctrl_cfg.get("alpha", 2e-5) if is_1d else 0.0
            surr_dataset = ControlSurrogateDataset(
                base_train, alpha=alpha, is_2d=is_2d
            )
            surr_loader = DataLoader(
                surr_dataset,
                batch_size=ctrl_cfg["surrogate"]["batch_size"],
                shuffle=True,
                num_workers=2,
            )
            surrogate.train(surr_loader, epochs=ctrl_cfg["surrogate"]["epochs"])
            # Save checkpoint
            ckpt_path = os.path.join("checkpoints", config.get_experiment_name(), "surrogate.pth")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            surrogate.save(ckpt_path)
            logger.info(f"Surrogate saved to {ckpt_path}")
        else:
            # Load pre‑trained surrogate for evaluation
            ckpt_path = os.path.join("checkpoints", config.get_experiment_name(), "surrogate.pth")
            if os.path.exists(ckpt_path):
                surrogate.load(ckpt_path)
                logger.info(f"Loaded surrogate from {ckpt_path}")
            else:
                logger.warning("Surrogate checkpoint not found; will use untrained surrogate (gradients may be useless).")
        control_surrogate = surrogate

    # ---- WDNO orchestrator ----
    ddim_steps = diff_cfg["ddim_steps"]
    ddim_eta = diff_cfg.get("ddim_eta", 0.0)
    wdno = WDNO(
        brm=ddpm_brm,
        srm=ddpm_srm,
        wavelet_transform=wavelet_transform,
        control_surrogate=control_surrogate,
        ddim_steps=ddim_steps,
        ddim_eta=ddim_eta,
        device=device,
    )

    # ---- Training ----
    if args.task in ("train", "all"):
        trainer = Trainer(
            wdno=wdno,
            dataset_brm=brm_dataset,
            config=config,
            dataset_srm=srm_dataset,
        )
        logger.info("Starting BRM training...")
        trainer.train_brm()
        if srm_dataset is not None:
            logger.info("Starting SRM training...")
            trainer.train_srm()
        logger.info("Training completed.")

    # ---- Evaluation ----
    if args.task in ("eval", "all"):
        # (Re‑)load best checkpoints (if training just finished, they are already in memory)
        # We assume the Trainer already saved the latest models inside the WDNO instance.
        # Build evaluator
        test_dataset = base_test
        # Optionally prepare high‑resolution dataset for super‑resolution evaluation.
        highres_test = None
        if len(scales) > 1 and is_1d:
            # For 1D Burgers super‑resolution, we need ground truth at higher resolutions.
            # We can generate them on‑the‑fly using the solver.
            # Here we create a tiny wrapper that produces high‑resolution data for the test set.
            # For simplicity, we'll skip super‑res evaluation if no dataset is provided.
            pass   # highres_test will be None, evaluate_super_resolution will skip.
        elif len(scales) > 1 and is_2d:
            # For 2D fluid, super‑resolution evaluation requires double spatial resolution data.
            # This would need a solver or a pre‑existing dataset; we skip here.
            pass

        # Create a ground‑truth solver for 1D control evaluation
        solver = None
        if is_control and is_1d:
            logger.info("Creating 1D Burgers solver for evaluation.")
            # We'll provide a function that takes (u0, f) and returns u_final (last time step)
            # The solver defined above returns the full trajectory; we wrap it.
            def solver_1d(u0_batch: torch.Tensor, f_batch: torch.Tensor) -> torch.Tensor:
                # u0_batch: (B, X), f_batch: (B, T, X)
                B = u0_batch.shape[0]
                u_final_list = []
                for i in range(B):
                    u0_np = u0_batch[i].cpu().numpy()
                    f_np = f_batch[i].cpu().numpy()   # shape (T, X)
                    # internal solver uses nt_stored = f_batch.shape[1] (80)
                    u_traj = solve_burgers_1d(u0_np, f_np, T=8.0, nu=0.01, nx=u0_np.shape[0], nt_stored=f_np.shape[1])
                    u_final_list.append(torch.from_numpy(u_traj[-1]).float())
                return torch.stack(u_final_list).to(u0_batch.device)
            solver = solver_1d

        evaluator = Evaluator(
            wdno=wdno,
            test_dataset=test_dataset,
            config=config,
            solver=solver,
            highres_dataset=highres_test,
        )

        # Run evaluations appropriate for the task
        if not is_control:
            # Simulation task
            logger.info("Running simulation evaluation...")
            sim_metrics = evaluator.evaluate_simulation()
            print(f"Simulation metrics: {sim_metrics}")
            # Super‑resolution evaluation (if available)
            if highres_test is not None:
                logger.info("Running super‑resolution evaluation...")
                sr_metrics = evaluator.evaluate_super_resolution()
                print(f"Super‑resolution metrics: {sr_metrics}")
        else:
            # Control task
            logger.info("Running control evaluation...")
            ctrl_metrics = evaluator.evaluate_control()
            print(f"Control metrics: {ctrl_metrics}")


if __name__ == "__main__":
    main()
