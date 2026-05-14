```python
## main.py
"""Main entry point for WDNO (Wavelet Diffusion Neural Operator) experiments.

This module ties together every component of the WDNO project: data generation,
wavelet transforms, diffusion model training, inference, super-resolution, and
evaluation. It parses CLI arguments, loads the config, and dispatches to
experiment-specific pipelines.

Usage:
    python main.py --experiment burgers --mode simulate --config config.yaml
    python main.py --experiment compressible_ns --mode simulate --skip_training --checkpoint checkpoints/burgers_brm_simulate.pt
    python main.py --experiment fluid_2d --mode control --device cuda
    python main.py --experiment burgers --mode ablation

Paper sources:
    - Experiment setup: Section 4, Appendices F, G, H
    - Evaluation protocol: Section 4 (MSE excluding IC), Appendix F.1 (solver for control)
    - Ablation studies: Section 4.7, Appendix C
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

# Project imports
from config import Config
from data.burgers_generator import BurgersGenerator
from data.fluid_2d_loader import Fluid2DLoader
from data.multi_resolution_dataset import MultiResolutionDataset
from data.pdebench_loader import PDEBenchLoader
from evaluation.evaluator import Evaluator
from evaluation.metrics import Metrics
from inference.control import Controller
from inference.simulate import Simulator
from inference.super_resolve import SuperResolver
from models.diffusion import Diffusion
from models.unet import UNet
from models.wdno_pipeline import WDNOPipeline
from training.trainer import Trainer
from utils.helpers import (
    load_checkpoint,
    make_dirs,
    save_checkpoint,
    set_seed,
)
from wavelet.wavelet_transform import WaveletTransform

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _setup_logging(results_dir: str) -> None:
    """Configure root logger with console and file handlers.

    Args:
        results_dir: Directory where the log file will be written.
    """
    make_dirs(results_dir)
    log_path = os.path.join(results_dir, "run.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
        force=True,
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace. CLI flags override config.yaml values.
    """
    parser = argparse.ArgumentParser(
        description="WDNO: Wavelet Diffusion Neural Operator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="burgers",
        choices=["burgers", "advection", "compressible_ns", "fluid_2d", "era5"],
        help="Experiment to run. Overrides config.experiment.name.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="simulate",
        choices=["simulate", "control", "super_resolve", "ablation"],
        help="Run mode. Overrides config.experiment.mode.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to config.yaml file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a pre-trained checkpoint. If provided, skips training.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Override config.data.data_path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda' or 'cpu'). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num_sr_levels",
        type=int,
        default=None,
        help="Override config.super_resolution.num_levels.",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        default=False,
        help="Skip training and load from checkpoint for inference/evaluation.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading with CLI overrides
# ---------------------------------------------------------------------------


def _load_config(args: argparse.Namespace) -> Config:
    """Load config from YAML and apply CLI overrides.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Fully populated Config instance with CLI overrides applied.

    Raises:
        FileNotFoundError: If the config YAML file does not exist.
    """
    if not os.path.exists(args.config):
        raise FileNotFoundError(
            f"Config file not found: '{args.config}'. "
            "Ensure config.yaml is in the current directory or specify --config."
        )

    with open(args.config, "r") as f:
        yaml_dict: Dict[str, Any] = yaml.safe_load(f)

    # Apply CLI overrides to the YAML dict before constructing Config
    if args.experiment is not None:
        yaml_dict.setdefault("experiment", {})["name"] = args.experiment
    if args.mode is not None:
        yaml_dict.setdefault("experiment", {})["mode"] = args.mode
    if args.seed is not None:
        yaml_dict.setdefault("experiment", {})["seed"] = args.seed
    if args.data_path is not None:
        yaml_dict.setdefault("data", {})["data_path"] = args.data_path
    if args.num_sr_levels is not None:
        yaml_dict.setdefault("super_resolution", {})["num_levels"] = args.num_sr_levels

    # Resolve device
    device: str
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = Config.from_dict(yaml_dict, device=device)
    return config


# ---------------------------------------------------------------------------
# Checkpoint path helpers
# ---------------------------------------------------------------------------


def _ckpt_path(config: Config, model_type: str, mode: str) -> str:
    """Build a standardized checkpoint file path.

    Args:
        config: Experiment configuration.
        model_type: Model identifier, e.g. 'brm_sim', 'brm_ctrl', 'srm'.
        mode: Mode string, e.g. 'simulate', 'control'.

    Returns:
        Full path string for the checkpoint file.
    """
    return os.path.join(
        config.checkpoint_dir,
        f"{config.experiment}_{model_type}_{mode}.pt",
    )


def _checkpoint_exists(path: str) -> bool:
    """Check if a checkpoint file exists on disk.

    Args:
        path: Full path to the checkpoint file.

    Returns:
        True if the file exists, False otherwise.
    """
    return os.path.isfile(path)


# ---------------------------------------------------------------------------
# Wavelet channel count computation
# ---------------------------------------------------------------------------


def _compute_wavelet_channels(
    wavelet_transform: WaveletTransform,
    sample_shape: Tuple[int, ...],
) -> int:
    """Compute the number of packed wavelet coefficient channels.

    Runs a dummy forward pass to determine the channel count after packing
    all coefficient sets along dim=1.

    Args:
        wavelet_transform: Configured WaveletTransform instance.
        sample_shape: Shape of a single sample (without batch dim).
            e.g. (81, 120) for 1D Burgers' or (32, 64, 64) for 2D fluid.

    Returns:
        Number of channels in the packed wavelet coefficient tensor.
        4 for 1D experiments (2D DWT), 8 for 2D experiments (3D DWT).
    """
    dummy = torch.zeros(1, *sample_shape)
    with torch.no_grad():
        W_dummy = wavelet_transform.forward(dummy)
    return int(W_dummy.shape[1])


# ---------------------------------------------------------------------------
# UNet construction helpers
# ---------------------------------------------------------------------------


def _build_unet_1d(
    config: Config,
    in_channels: int,
    out_channels: int,
    cond_channels: int,
) -> UNet:
    """Build a 1D U-Net (2D convolutions on time×space wavelet coefficients).

    Args:
        config: Experiment configuration. Reads unet_1d.* fields.
        in_channels: Number of noisy input channels (wavelet coeff sets).
        out_channels: Number of output channels (predicted noise).
        cond_channels: Number of conditioning channels.

    Returns:
        Configured UNet instance in mode='1d'.
    """
    return UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        cond_channels=cond_channels,
        init_dim=config.unet_init_dim,
        dim_mults=tuple(config.unet_dim_mults),
        resnet_groups=config.unet_resnet_groups,
        attn_heads=config.unet_attn_heads,
        attn_head_dim=config.unet_attn_head_dim,
        mode="1d",
    )


def _build_unet_2d(
    config: Config,
    in_channels: int,
    out_channels: int,
    cond_channels: int,
) -> UNet:
    """Build a 2D U-Net (3D convolutions on time×height×width wavelet coefficients).

    Args:
        config: Experiment configuration. Reads unet_3d.* fields.
        in_channels: Number of noisy input channels.
        out_channels: Number of output channels.
        cond_channels: Number of conditioning channels.

    Returns:
        Configured UNet instance in mode='2d'.
    """
    return UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        cond_channels=cond_channels,
        init_dim=config.unet_init_dim,
        dim_mults=tuple(config.unet_dim_mults),
        resnet_groups=config.unet_resnet_groups,
        attn_heads=config.unet_attn_heads,
        attn_head_dim=config.unet_attn_head_dim,
        mode="2d",
        conv3d_kernel=tuple(config.conv3d_kernel),
        conv3d_padding=tuple(config.conv3d_padding),
        conv3d_stride=tuple(config.conv3d_stride),
        downsample_kernel=tuple(config.downsample_kernel),
        downsample_padding=tuple(config.downsample_padding),
        downsample_stride=tuple(config.downsample_stride),
        upsample_kernel=tuple(config.upsample_kernel),
        upsample_padding=tuple(config.upsample_padding),
        upsample_stride=tuple(config.upsample_stride),
    )


def _build_diffusion(
    model: UNet,
    config: Config,
) -> Diffusion:
    """Wrap a UNet in a Diffusion instance with the configured noise schedule.

    Args:
        model: UNet denoising network.
        config: Experiment configuration. Reads diffusion.* fields.

    Returns:
        Configured Diffusion instance.
    """
    return Diffusion(
        model=model,
        timesteps=config.num_diffusion_timesteps,
        beta_schedule=config.beta_schedule,
        device=config.device,
    )


# ---------------------------------------------------------------------------
# DataLoader construction helpers
# ---------------------------------------------------------------------------


def _make_dataloader(
    tensors: Tuple[torch.Tensor, ...],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Wrap tensors in a TensorDataset and return a DataLoader.

    Args:
        tensors: Tuple of tensors with matching first dimension (N samples).
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle samples each epoch.
        num_workers: Number of DataLoader worker processes.

    Returns:
        Configured DataLoader.
    """
    dataset = TensorDataset(*tensors)
    pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Wavelet pre-transformation helpers
# ---------------------------------------------------------------------------


def _pretransform_1d_state(
    wt: WaveletTransform,
    u: torch.Tensor,
    batch_size: int = 500,
) -> torch.Tensor:
    """Apply 2D DWT to 1D PDE state trajectories in mini-batches.

    Args:
        wt: WaveletTransform configured for 1D PDEs (spatial_dim=1).
        u: State tensor of shape [N, T, X]. float32.
        batch_size: Mini-batch size for memory-efficient processing.

    Returns:
        Packed wavelet coefficients of shape [N, 4, T_c, X_c]. float32.
    """
    N = u.shape[0]
    results: List[torch.Tensor] = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        with torch.no_grad():
            W = wt.forward(u[start:end])
        results.append(W.cpu())
    return torch.cat(results, dim=0)


def _pretransform_1d_condition(
    wt: WaveletTransform,
    u0: torch.Tensor,
    f: torch.Tensor,
    batch_size: int = 500,
) -> torch.Tensor:
    """Prepare conditioning wavelet coefficients for 1D PDE experiments.

    Applies 2D DWT to force f and 1D DWT to initial condition u0, tiles
    u0 coefficients along the time dimension, then concatenates channel-wise.

    Paper Appendix F.3: "we take the 1D wavelet transform, repeat the
    coefficients, and then concatenate them with the 2D coefficients."

    Args:
        wt: WaveletTransform configured for 1D PDEs (spatial_dim=1).
        u0: Initial conditions of shape [N, X]. float32.
        f: Force terms of shape [N, T, X]. float32.
        batch_size: Mini-batch size.

    Returns:
        Conditioning tensor of shape [N, C_cond, T_c, X_c]. float32.
        C_cond = 4 (from f DWT) + 2 (from u0 1D DWT tiled) = 6.
    """
    N = f.shape[0]
    W_f_list: List[torch.Tensor] = []
    W_u0_list: List[torch.Tensor] = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        f_batch = f[start:end]
        u0_batch = u0[start:end]

        with torch.no_grad():
            # 2D DWT on force: [B, T, X] → [B, 4, T_c, X_c]
            W_f_batch = wt.forward(f_batch)

            # 1D DWT on u0: [B, X] → [B, 2, X_c]
            # Use pytorch_wavelets DWTForward with J=1 on 1D data
            # Treat u0 as [B, 1, 1, X] for 2D DWT, then extract coarse + detail
            u0_4d = u0_batch.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, X]
            try:
                from pytorch_wavelets import DWTForward
                dwt_1d = DWTForward(J=1, wave=wt.wavelet, mode=wt.mode)
                dwt_1d = dwt_1d.to(u0_batch.device)
                yl_1d, yh_1d = dwt_1d(u0_4d)
                # yl_1d: [B, 1, 1, X_c], yh_1d[0]: [B, 1, 3, 1, X_c]
                cA_1d = yl_1d.squeeze(1).squeeze(1)  # [B, X_c]
                cD_1d = yh_1d[0][:, 0, 0, 0, :]      # [B, X_c] (first detail subband)
                W_u0_batch = torch.stack([cA_1d, cD_1d], dim=1)  # [B, 2, X_c]
            except Exception:
                # Fallback: use simple downsampling as approximation
                X_c = W_f_batch.shape[3]
                W_u0_batch = torch.stack([
                    u0_batch[:, ::2][:, :X_c],
                    u0_batch[:, 1::2][:, :X_c],
                ], dim=1)  # [B, 2, X_c]

            # Tile u0 coefficients along time dimension
            T_c = W_f_batch.shape[2]
            X_c = W_f_batch.shape[3]
            # [B, 2, X_c] → [B, 2, T_c, X_c]
            W_u0_tiled = W_u0_batch.unsqueeze(2).expand(-1, -1, T_c, -1).contiguous()

            # Concatenate: [B, 4, T_c, X_c] + [B, 2, T_c, X_c] → [B, 6, T_c, X_c]
            W_cond_batch = torch.cat([W_f_batch, W_u0_tiled], dim=1)

        W_f_list.append(W_f_batch.cpu())
        W_u0_list.append(W_cond_batch.cpu())

    return torch.cat(W_u0_list, dim=0)


# ---------------------------------------------------------------------------
# Burgers' solver function factory
# ---------------------------------------------------------------------------


def _make_burgers_solver_fn(
    generator: BurgersGenerator,
    u0_dataset: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a ground-truth Burgers' solver callable for control evaluation.

    The returned function takes a force sequence f [B, T, X] and returns
    the final state u_T [B, X] using the finite difference solver.

    Paper Appendix F.1: "the state deviation in our reported evaluation
    metric I is always based on the output u(T,x) of the ground-truth
    solver given the control force f(t,x)."

    Args:
        generator: BurgersGenerator instance with the configured solver.
        u0_dataset: Initial conditions for the test set, shape [N, X].
            Used to provide u0 to the solver for each test sample.

    Returns:
        Callable solver_fn(f: Tensor) -> Tensor where:
            f: [B, T, X] control force
            returns: u_T [B, X] final state from ground-truth solver
    """
    _u0 = u0_dataset.clone()

    def solver_fn(f: torch.Tensor) -> torch.Tensor:
        """Run ground-truth Burgers' solver.

        Args:
            f: Control force of shape [B, T, X]. float32.

        Returns:
            Final state u_T of shape [B, X]. float32.
        """
        B = f.shape[0]
        # Use the first B initial conditions from the test set
        u0_batch = _u0[:B].to(f.device, dtype=torch.float64)
        f_batch = f.to(dtype=torch.float64)

        with torch.no_grad():
            # _solve_pde returns [B, T+1, X] including t=0
            trajectory = generator._solve_pde(u0_batch, f_batch)
            # Extract final state: [B, X]
            u_T = trajectory[:, -1, :].float()

        return u_T

    return solver_fn


# ---------------------------------------------------------------------------
# Results printing
# ---------------------------------------------------------------------------


def _print_results_table(results: Dict[str, Any]) -> None:
    """Print evaluation results in a format matching the paper's tables.

    Args:
        results: Dict mapping result category names to metric dicts.
            Keys like 'simulation', 'control', 'super_resolution', etc.
    """
    separator = "=" * 70
    print(f"\n{separator}")
    print("WDNO EVALUATION RESULTS")
    print(separator)

    # --- Simulation results (Table 1) ---
    if "simulation" in results:
        sim = results["simulation"]
        print("\n[Simulation Metrics] (Table 1)")
        print(f"  MSE:         {sim.get('mse', float('nan')):.6e}")
        print(f"  MAE:         {sim.get('mae', float('nan')):.6e}")
        print(f"  L_inf:       {sim.get('l_inf', float('nan')):.6e}")
        print(f"  Relative L2: {sim.get('relative_l2', float('nan')):.6e}")
        print(f"  N samples:   {sim.get('n_samples', 0)}")

    # --- Control results (Table 2a / 2b) ---
    if "control" in results:
        ctrl = results["control"]
        print("\n[Control Metrics] (Table 2)")
        print(f"  Mean J:   {ctrl.get('mean_J', float('nan')):.6f}")
        if "std_J" in ctrl:
            print(f"  Std J:    {ctrl.get('std_J', float('nan')):.6f}")
        print(f"  Median J: {ctrl.get('median_J', float('nan')):.6f}")
        print(f"  N samples: {ctrl.get('n_samples', 0)}")

    # --- Super-resolution results (Table 16 / 17) ---
    if "super_resolution" in results:
        sr = results["super_resolution"]
        print("\n[Super-Resolution MSE] (Table 16/17)")
        for level_key, interp_dict in sorted(sr.items()):
            if isinstance(interp_dict, dict):
                for mode, mse_val in interp_dict.items():
                    print(f"  Level {level_key} ({mode}): {mse_val:.6e}")
            else:
                print(f"  Level {level_key}: {interp_dict:.6e}")

    # --- Ablation results ---
    for key, val in results.items():
        if key in ("simulation", "control", "super_resolution"):
            continue
        print(f"\n[{key}]")
        if isinstance(val, dict):
            for k, v in val.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {val}")

    print(f"\n{separator}\n")


# ---------------------------------------------------------------------------
# Experiment: 1D Burgers' equation
# ---------------------------------------------------------------------------


def _run_burgers(
    config: Config,
    skip_training: bool = False,
    checkpoint_path: Optional[str] = None,
) -> None:
    """Run the full WDNO pipeline for the 1D Burgers' equation experiment.

    Covers simulation, control, and zero-shot super-resolution as described
    in paper Section 4.1 and Appendix F.

    Args:
        config: Experiment configuration.
        skip_training: If True, load checkpoints instead of training.
        checkpoint_path: Optional explicit checkpoint path to load.
    """
    logger.info("=" * 60)
    logger.info("Running 1D Burgers' equation experiment")
    logger.info("Mode: %s", config.mode)
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Step 1: Generate / Load Dataset
    # -----------------------------------------------------------------------
    generator = BurgersGenerator(
        nu=config.nu,
        T=config.T,
        nx_fine=config.nx_fine,
        nt_fine=config.nt_fine,
        nx_coarse=config.nx_coarse,
        nt_coarse=config.nt_coarse,
        num_train=config.num_train,
        num_test_control=config.num_test_control,
        num_test_sr_base=config.num_test_sr_base,
        num_test_sr_levels=config.num_test_sr_levels,
        device=config.device,
        solver_batch_size=50,
    )

    train_path = os.path.join(config.data_path, "burgers_train.pt")
    ctrl_test_path = os.path.join(config.data_path, "burgers_test_control.pt")
    sr_base_path = os.path.join(config.data_path, "burgers_test_sr_base.pt")

    if _checkpoint_exists(train_path):
        logger.info("Loading existing Burgers' training dataset from %s", train_path)
        train_dataset = generator.load_dataset(train_path)
    else:
        logger.info("Generating Burgers' training dataset (%d trajectories)...", config.num_train)
        train_dataset = generator.generate_dataset(save_path=train_path)

    # Extract tensors
    u_train: torch.Tensor = train_dataset["train"]["u"].float()   # [N, 81, 120]
    f_train: torch.Tensor = train_dataset["train"]["f"].float()   # [N, 80, 120]
    u0_train: torch.Tensor = train_dataset["train"]["u0"].float() # [N, 120]

    # Test data for simulation evaluation
    u_test: torch.Tensor = train_dataset.get("test_sr", {}).get(
        "u", u_train[-200:]
    ).float()
    f_test: torch.Tensor = train_dataset.get("test_sr", {}).get(
        "f", f_train[-200:]
    ).float()
    u0_test: torch.Tensor = train_dataset.get("test_sr", {}).get(
        "u0", u0_train[-200:]
    ).float()

    # Control test data
    u_ctrl_test: torch.Tensor = train_dataset.get("test_control", {}).get(
        "u", u_train[:config.num_test_control]
    ).float()
    f_ctrl_test: torch.Tensor = train_dataset.get("test_control", {}).get(
        "f", f_train[:config.num_test_control]
    ).float()
    u0_ctrl_test: torch.Tensor = train_dataset.get("test_control", {}).get(
        "u0", u0_train[:config.num_test_control]
    ).float()
    u_star_ctrl: torch.Tensor = train_dataset.get("test_control", {}).get(
        "u_star", u_train[:config.num_test_control, -1, :]
    ).float()  # [N_ctrl, 120]

    logger.info(
        "Dataset loaded: train=%d, test_sim=%d, test_ctrl=%d",
        u_train.shape[0], u_test.shape[0], u_ctrl_test.shape[0],
    )

    # -----------------------------------------------------------------------
    # Step 2: Initialize WaveletTransform
    # -----------------------------------------------------------------------
    wavelet_transform = WaveletTransform(
        wavelet=config.wavelet_type,    # 'bior2.4'
        mode=config.wavelet_mode,       # 'periodization'
        level=config.wavelet_level,     # 1
        spatial_dim=config.spatial_dim, # 1
    )

    # -----------------------------------------------------------------------
    # Step 3: Verify Reconstruction Error
    # -----------------------------------------------------------------------
    sample_for_verify = u_train[:10].float()
    recon_error = wavelet_transform.verify_reconstruction(sample_for_verify)
    logger.info("Wavelet reconstruction error: %.2e (expected ~1e-7)", recon_error)
    if recon_error > 1e-4:
        logger.warning(
            "Reconstruction error %.2e is higher than expected (~1e-7). "
            "Check wavelet configuration.",
            recon_error,
        )

    # -----------------------------------------------------------------------
    # Step 4: Compute channel counts and build UNets
    # -----------------------------------------------------------------------
    # Wavelet channels for state u (shape [81, 120] → 4 coeff sets)
    u_wavelet_channels = _compute_wavelet_channels(wavelet_transform, (81, 120))
    # Wavelet channels for force f (shape [80, 120] → 4 coeff sets)
    f_wavelet_channels = _compute_wavelet_channels(wavelet_transform, (80, 120))
    # Conditioning channels: f (4) + u0 tiled (2) = 6
    cond_channels_sim = f_wavelet_channels + 2  # u0 contributes 2 channels (cA, cD)
    # For control: conditioning is u0 + u_star (both 1D → 2 channels each = 4 total)
    cond_channels_ctrl = 4  # u0 (2 channels) + u_star (2 channels)

    logger.info(
        "Channel counts: u_wavelet=%d, f_wavelet=%d, cond_sim=%d, cond_ctrl=%d",
        u_wavelet_channels, f_wavelet_channels, cond_channels_sim, cond_channels_ctrl,
    )

    # Build simulation BRM UNet (predicts state u conditioned on u0 + f)
    unet_sim = _build_unet_1d(
        config=config,
        in_channels=u_wavelet_channels,
        out_channels=u_wavelet_channels,
        cond_channels=cond_channels_sim,
    )
    diffusion_sim = _build_diffusion(unet_sim, config)

    # Build control BRM UNet (predicts force f conditioned on u0 + u_star)
    unet_ctrl = _build_unet_1d(
        config=config,
        in_channels=f_wavelet_channels,
        out_channels=f_wavelet_channels,
        cond_channels=cond_channels_ctrl,
    )
    diffusion_ctrl = _build_diffusion(unet_ctrl, config)

    logger.info(
        "UNet parameters: sim=%d, ctrl=%d",
        sum(p.numel() for p in unet_sim.parameters()),
        sum(p.numel() for p in unet_ctrl.parameters()),
    )

    # -----------------------------------------------------------------------
    # Step 5: Pre-transform training data
    # -----------------------------------------------------------------------
    logger.info("Pre-transform