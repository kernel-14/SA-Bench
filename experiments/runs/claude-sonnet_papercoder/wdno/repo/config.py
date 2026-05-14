```python
## config.py
"""Central configuration module for WDNO (Wavelet Diffusion Neural Operator).

This module defines the Config dataclass that centralizes all hyperparameters
from the paper. Every other module imports Config from here.

Paper sources:
    - Wavelet config: Appendix A, F.3, H.2
    - U-Net architecture: Table 18, 19, 20
    - Training: Table 18, 19, 20, Appendix C.6
    - Inference: Table 18, 19, 20
    - Data generation: Appendix F.1, F.2, G.1
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Wavelet configuration lookup (paper: Appendix A, F.3, H.2)
# ---------------------------------------------------------------------------

_WAVELET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "burgers": {
        "wavelet_type": "bior2.4",
        "padding_mode": "periodization",
        "transform_dim": 2,
        "level": 1,
        "library": "pytorch_wavelets",
    },
    "advection": {
        "wavelet_type": "bior2.4",
        "padding_mode": "periodization",
        "transform_dim": 2,
        "level": 1,
        "library": "pytorch_wavelets",
    },
    "compressible_ns": {
        "wavelet_type": "bior2.4",
        "padding_mode": "periodization",
        "transform_dim": 2,
        "level": 1,
        "library": "pytorch_wavelets",
    },
    "fluid_2d": {
        "wavelet_type": "bior1.3",
        "padding_mode": "zero",
        "transform_dim": 3,
        "level": 1,
        "library": "ptwt",
    },
    "era5": {
        "wavelet_type": "bior1.3",
        "padding_mode": "zero",
        "transform_dim": 3,
        "level": 1,
        "library": "ptwt",
    },
}

_VALID_EXPERIMENTS: Tuple[str, ...] = (
    "burgers",
    "advection",
    "compressible_ns",
    "fluid_2d",
    "era5",
)

_VALID_MODES: Tuple[str, ...] = (
    "simulate",
    "control",
    "super_resolve",
    "ablation",
)


@dataclass
class Config:
    """Flat configuration dataclass for WDNO experiments.

    All fields are populated from config.yaml via from_dict(). Experiment-
    specific values (wavelet type, DDIM steps, guidance lambda, etc.) are
    resolved based on the experiment name during construction.

    Attributes:
        experiment: Experiment name, one of 'burgers', 'advection',
            'compressible_ns', 'fluid_2d', 'era5'.
        mode: Run mode, one of 'simulate', 'control', 'super_resolve',
            'ablation'.
        spatial_dim: Spatial dimensionality of the PDE (1 or 2). Drives
            choice of 2D vs 3D wavelet transform and U-Net.
        seed: Random seed for reproducibility.
        device: Compute device ('cuda' or 'cpu').

        wavelet_type: Wavelet basis, e.g. 'bior2.4' or 'bior1.3'.
        wavelet_mode: Padding mode, 'periodization' or 'zero'.
        wavelet_level: Decomposition level (always 1, l_0=L finest level).
        wavelet_transform_dim: DWT dimensionality (2 for 1D PDEs, 3 for 2D).
        wavelet_library: Backend library ('pytorch_wavelets' or 'ptwt').

        num_diffusion_timesteps: Total DDPM training timesteps (K=1000).
        beta_schedule: Noise schedule type ('cosine' or 'linear').
        cfg_dropout_prob: Classifier-free guidance condition dropout rate.
        cfg_weight: CFG combination weight omega.

        ddim_steps: DDIM inference sampling steps (experiment-specific).
        ddim_eta: DDIM stochasticity parameter eta.
        guidance_lambda: Control guidance weight lambda.
        guidance_schedule: Schedule type for guidance weight ('cosine').

        unet_init_dim: Initial channel dimension for U-Net.
        unet_dim_mults: Channel multipliers per downsampling level.
        unet_resnet_groups: Group norm groups in ResNet blocks.
        unet_attn_heads: Number of attention heads.
        unet_attn_head_dim: Hidden dimension per attention head.
        unet_n_downsample: Number of downsampling/upsampling levels.
        use_dual_unet: Whether to use two separate U-Nets (1D experiments).

        conv3d_kernel: 3D conv kernel size [T, H, W] (2D experiments).
        conv3d_padding: 3D conv padding [T, H, W].
        conv3d_stride: 3D conv stride [T, H, W].
        downsample_kernel: Downsampling conv kernel [T, H, W].
        downsample_padding: Downsampling conv padding [T, H, W].
        downsample_stride: Downsampling conv stride [T, H, W].
        upsample_kernel: Upsampling conv kernel [T, H, W].
        upsample_padding: Upsampling conv padding [T, H, W].
        upsample_stride: Upsampling conv stride [T, H, W].

        batch_size: Training batch size.
        lr: Learning rate.
        train_steps: Total gradient update steps.
        lr_scheduler: LR scheduler type ('cosine_annealing').
        num_gpus: Number of GPUs for training.

        num_sr_levels: Number of super-resolution levels at inference.
        eval_interp_modes: Interpolation modes for SR evaluation.

        data_path: Root path for raw data files.
        checkpoint_dir: Directory for saving/loading checkpoints.
        results_dir: Directory for saving evaluation results.

        nu: Burgers' equation diffusion coefficient (0.01).
        T: Burgers' equation time horizon (8.0).
        nx_fine: Fine spatial grid size for Burgers' solver (1920).
        nt_fine: Fine temporal grid size for Burgers' solver (76800).
        nx_coarse: Coarse spatial grid size stored in dataset (120).
        nt_coarse: Coarse temporal steps stored in dataset (80).
        num_train: Number of training trajectories.
        num_test_control: Number of control test trajectories.
        num_test_sr_base: Number of SR test trajectories at base resolution.
        num_test_sr_levels: Number of shared SR test trajectories.
        control_alpha: Weight alpha in control objective I.
        solver_nx: Solver internal spatial grid size for evaluation.
        solver_nt: Solver internal temporal grid size for evaluation.

        compressible_ns_hdf5_filename: PDEBench HDF5 filename for NS data.
        compressible_ns_eta: Shear viscosity (1e-8).
        compressible_ns_zeta: Bulk viscosity (1e-8).
        compressible_ns_num_variables: Number of output variables (3).

        fluid_2d_nt: Number of timesteps for 2D fluid (32).
        fluid_2d_nx: Spatial grid width (64).
        fluid_2d_ny: Spatial grid height (64).
        fluid_2d_num_control_vars: Control variables per timestep (3584).

        ablation_noise_scale_factors: Noise scale factors for noise ablation.
        ablation_data_size_fractions: Dataset size fractions for size ablation.
        ablation_train_size: Full training set size used in ablation (9000).
        ablation_ddim_step_values: DDIM step counts for sensitivity analysis.
        ablation_ddim_eta_values: DDIM eta values for sensitivity analysis.
        ablation_wavelet_types: Wavelet types to compare in ablation.
        ablation_control_noise_prob: Noise probability for control robustness.
    """

    # --- Experiment identification ---
    experiment: str = "burgers"
    mode: str = "simulate"
    spatial_dim: int = 1
    seed: int = 42
    device: str = "cuda"

    # --- Wavelet transform (paper: Appendix A, F.3, H.2) ---
    wavelet_type: str = "bior2.4"
    wavelet_mode: str = "periodization"
    wavelet_level: int = 1
    wavelet_transform_dim: int = 2
    wavelet_library: str = "pytorch_wavelets"

    # --- Diffusion model (paper: Section 2.2) ---
    num_diffusion_timesteps: int = 1000
    beta_schedule: str = "cosine"
    cfg_dropout_prob: float = 0.1
    cfg_weight: float = 1.0

    # --- Inference / DDIM (paper: Table 18, 19, 20) ---
    ddim_steps: int = 50
    ddim_eta: float = 1.0
    guidance_lambda: float = 120000.0
    guidance_schedule: str = "cosine"

    # --- 1D U-Net architecture (paper: Table 18, 19) ---
    unet_init_dim: int = 128
    unet_dim_mults: Tuple[int, ...] = field(default_factory=lambda: (1, 2, 4, 8))
    unet_resnet_groups: int = 8
    unet_attn_heads: int = 4
    unet_attn_head_dim: int = 32
    unet_n_downsample: int = 4
    use_dual_unet: bool = True

    # --- 3D U-Net architecture (paper: Table 20, 2D experiments only) ---
    conv3d_kernel: List[int] = field(default_factory=lambda: [3, 3, 3])
    conv3d_padding: List[int] = field(default_factory=lambda: [1, 1, 1])
    conv3d_stride: List[int] = field(default_factory=lambda: [1, 1, 1])
    downsample_kernel: List[int] = field(default_factory=lambda: [1, 4, 4])
    downsample_padding: List[int] = field(default_factory=lambda: [0, 1, 1])
    downsample_stride: List[int] = field(default_factory=lambda: [1, 2, 2])
    upsample_kernel: List[int] = field(default_factory=lambda: [1, 4, 4])
    upsample_padding: List[int] = field(default_factory=lambda: [0, 1, 1])
    upsample_stride: List[int] = field(default_factory=lambda: [1, 2, 2])

    # --- Training (paper: Table 18, 19, 20, Appendix C.6) ---
    batch_size: int = 16
    lr: float = 1e-4
    train_steps: int = 190000
    lr_scheduler: str = "cosine_annealing"
    num_gpus: int = 1

    # --- Super-resolution (paper: Section 4.6, Table 16, 17) ---
    num_sr_levels: int = 0
    eval_interp_modes: List[str] = field(default_factory=lambda: ["linear", "nearest"])

    # --- Paths ---
    data_path: str = "./data/raw"
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"

    # --- Burgers' equation data (paper: Appendix F.1, F.2) ---
    nu: float = 0.01
    T: float = 8.0
    nx_fine: int = 1920
    nt_fine: int = 76800
    nx_coarse: int = 120
    nt_coarse: int = 80
    num_train: int = 40000
    num_test_control: int = 50
    num_test_sr_base: int = 2000
    num_test_sr_levels: int = 100
    control_alpha: float = 0.1
    solver_nx: int = 1920
    solver_nt: int = 76800

    # --- Compressible NS data (paper: Appendix G.1) ---
    compressible_ns_hdf5_filename: str = (
        "1D_CFD_Shock_Eta1.e-8_Zeta1.e-8_trans_Train.hdf5"
    )
    compressible_ns_eta: float = 1e-8
    compressible_ns_zeta: float = 1e-8
    compressible_ns_num_variables: int = 3

    # --- 2D fluid data (paper: Section 4.4, Appendix H.1) ---
    fluid_2d_nt: int = 32
    fluid_2d_nx: int = 64
    fluid_2d_ny: int = 64
    fluid_2d_num_control_vars: int = 3584

    # --- Ablation study (paper: Section 4.7, Appendix C) ---
    ablation_noise_scale_factors: List[float] = field(
        default_factory=lambda: [0.0001, 0.001, 0.01]
    )
    ablation_data_size_fractions: List[float] = field(
        default_factory=lambda: [0.2, 0.4, 0.6, 0.8]
    )
    ablation_train_size: int = 9000
    ablation_ddim_step_values: List[int] = field(
        default_factory=lambda: [20, 40, 50, 100, 200]
    )
    ablation_ddim_eta_values: List[float] = field(
        default_factory=lambda: [0.2, 0.5, 0.8, 1.0]
    )
    ablation_wavelet_types: List[str] = field(
        default_factory=lambda: ["bior1.3", "bior2.4", "db4", "sym4"]
    )
    ablation_control_noise_prob: float = 0.1

    # -----------------------------------------------------------------------
    # Class methods
    # -----------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        d: Dict[str, Any],
        device: Optional[str] = None,
    ) -> "Config":
        """Construct a Config from a parsed YAML dictionary.

        Reads the experiment name from ``d['experiment']['name']`` and
        resolves all experiment-specific fields (wavelet, inference, training,
        U-Net architecture, data) from the corresponding YAML sub-sections.

        Args:
            d: Full parsed YAML dictionary (from ``yaml.safe_load``).
            device: Optional device override. If None, auto-detects CUDA.

        Returns:
            Fully populated Config instance.

        Raises:
            ValueError: If the experiment name is not recognised.
        """
        # --- Resolve device ---
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        # --- Experiment identification ---
        exp_section: Dict[str, Any] = d.get("experiment", {})
        experiment: str = str(exp_section.get("name", "burgers"))
        mode: str = str(exp_section.get("mode", "simulate"))
        spatial_dim: int = int(exp_section.get("spatial_dim", 1))
        seed: int = int(exp_section.get("seed", 42))

        if experiment not in _VALID_EXPERIMENTS:
            raise ValueError(
                f"Unknown experiment '{experiment}'. "
                f"Must be one of {_VALID_EXPERIMENTS}."
            )
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Unknown mode '{mode}'. Must be one of {_VALID_MODES}."
            )

        # --- Wavelet configuration ---
        wavelet_cfg: Dict[str, Any] = cls.get_wavelet_config(experiment)
        wavelet_type: str = wavelet_cfg["wavelet_type"]
        wavelet_mode: str = wavelet_cfg["padding_mode"]
        wavelet_level: int = wavelet_cfg["level"]
        wavelet_transform_dim: int = wavelet_cfg["transform_dim"]
        wavelet_library: str = wavelet_cfg["library"]

        # --- Diffusion model ---
        diff_section: Dict[str, Any] = d.get("diffusion", {})
        num_diffusion_timesteps: int = int(diff_section.get("num_timesteps", 1000))
        beta_schedule: str = str(diff_section.get("beta_schedule", "cosine"))
        cfg_dropout_prob: float = float(diff_section.get("cfg_dropout_prob", 0.1))
        cfg_weight: float = float(diff_section.get("cfg_weight", 1.0))

        # --- Inference (experiment-specific) ---
        infer_section: Dict[str, Any] = d.get("inference", {})
        infer_exp: Dict[str, Any] = infer_section.get(experiment, {})
        ddim_steps: int = int(infer_exp.get("ddim_steps", 50))
        ddim_eta: float = float(infer_exp.get("ddim_eta", 1.0))
        guidance_lambda: float = float(infer_exp.get("guidance_lambda", 120000.0))
        guidance_schedule: str = str(infer_exp.get("guidance_schedule", "cosine"))
        cfg_weight = float(infer_exp.get("cfg_weight", cfg_weight))

        # --- U-Net architecture ---
        if spatial_dim == 1:
            unet_section: Dict[str, Any] = d.get("unet_1d", {})
            unet_init_dim: int = int(unet_section.get("init_dim", 128))
            unet_dim_mults_raw: List[int] = list(
                unet_section.get("dim_mults", [1, 2, 4, 8])
            )
            unet_dim_mults: Tuple[int, ...] = tuple(unet_dim_mults_raw)
            unet_resnet_groups: int = int(unet_section.get("resnet_block_groups", 8))
            unet_attn_heads: int = int(unet_section.get("attn_heads", 4))
            unet_attn_head_dim: int = int(unet_section.get("attn_hidden_dim", 32))
            unet_n_downsample: int = int(
                unet_section.get("n_downsample_layers", 4)
            )
            use_dual_unet: bool = bool(unet_section.get("use_dual_unet", True))
            # 3D U-Net fields use defaults for 1D experiments
            conv3d_kernel: List[int] = [3, 3, 3]
            conv3d_padding: List[int] = [1, 1, 1]
            conv3d_stride: List[int] = [1, 1, 1]
            downsample_kernel: List[int] = [1, 4, 4]
            downsample_padding: List[int] = [0, 1, 1]
            downsample_stride: List[int] = [1, 2, 2]
            upsample_kernel: List[int] = [1, 4, 4]
            upsample_padding: List[int] = [0, 1, 1]
            upsample_stride: List[int] = [1, 2, 2]
        else:
            # 2D experiments use 3D U-Net
            unet_3d_section: Dict[str, Any] = d.get("unet_3d", {})
            unet_init_dim = 128  # not explicitly stated for 3D; use same default
            unet_dim_mults = (1, 2, 4, 8)
            unet_resnet_groups = 8
            unet_attn_heads = int(unet_3d_section.get("attn_heads", 4))
            unet_attn_head_dim = 32
            unet_n_downsample = 4
            use_dual_unet = False
            conv3d_kernel = list(unet_3d_section.get("conv3d_kernel", [3, 3, 3]))
            conv3d_padding = list(
                unet_3d_section.get("conv3d_padding", [1, 1, 1])
            )
            conv3d_stride = list(unet_3d_section.get("conv3d_stride", [1, 1, 1]))
            downsample_kernel = list(
                unet_3d_section.get("downsample_kernel", [1, 4, 4])
            )
            downsample_padding = list(
                unet_3d_section.get("downsample_padding", [0, 1, 1])
            )
            downsample_stride = list(
                unet_3d_section.get("downsample_stride", [1, 2, 2])
            )
            upsample_kernel = list(
                unet_3d_section.get("upsample_kernel", [1, 4, 4])
            )
            upsample_padding = list(
                unet_3d_section.get("upsample_padding", [0, 1, 1])
            )
            upsample_stride = list(
                unet_3d_section.get("upsample_stride", [1, 2, 2])
            )

        # --- Training (experiment-specific) ---
        train_section: Dict[str, Any] = d.get("training", {})
        train_exp: Dict[str, Any] = train_section.get(experiment, {})
        batch_size: int = int(train_exp.get("batch_size", 16))
        lr: float = float(train_exp.get("learning_rate", 1e-4))
        train_steps: int = int(train_exp.get("train_steps", 190000))
        lr_scheduler: str = str(train_exp.get("lr_scheduler", "cosine_annealing"))
        num_gpus: int = int(train_exp.get("num_gpus", 1))

        # --- Super-resolution ---
        sr_section: Dict[str, Any] = d.get("super_resolution", {})
        num_sr_levels: int = int(sr_section.get("num_levels", 0))
        eval_interp_modes: List[str] = list(
            sr_section.get("eval_interp_modes", ["linear", "nearest"])
        )

        # --- Paths ---
        data_section: Dict[str, Any] = d.get("data", {})
        data_path: str = str(data_section.get("data_path", "./data/raw"))
        checkpoint_dir: str = str(
            data_section.get("checkpoint_dir", "./checkpoints")
        )
        results_dir: str = str(data_section.get("results_dir", "./results"))

        # --- Burgers' equation data ---
        burgers_data: Dict[str, Any] = data_section.get("burgers", {})
        nu: float = float(burgers_data.get("nu", 0.01))
        T: float = float(burgers_data.get("T", 8.0))
        nx_fine: int = int(burgers_data.get("nx_fine", 1920))
        nt_fine: int = int(burgers_data.get("nt_fine", 76800))
        nx_coarse: int = int(burgers_data.get("nx_coarse", 120))
        nt_coarse: int = int(burgers_data.get("nt_coarse", 80))
        num_train: int = int(burgers_data.get("num_train", 40000))
        num_test_control: int = int(burgers_data.get("num_test_control", 50))
        num_test_sr_base: int = int(burgers_data.get("num_test_sr_base", 2000))
        num_test_sr_levels: int = int(burgers_data.get("num_test_sr_levels", 100))
        control_alpha: float = float(burgers_data.get("control_alpha", 0.1))
        solver_nx: int = int(burgers_data.get("solver_nx", 1920))
        solver_nt: int = int(burgers_data.get("solver_nt", 76800))

        # --- Compressible NS data ---
        cns_data: Dict[str, Any] = data_section.get("compressible_ns", {})
        compressible_ns_hdf5_filename: str = str(
            cns_data.get(
                "hdf5_filename",
                "1D_CFD_Shock_Eta1.e-8_Zeta1.e-8_trans_Train.hdf5",
            )
        )
        compressible_ns_eta: float = float(cns_data.get("eta", 1e-8))
        compressible_ns_zeta: float = float(cns_data.get("zeta", 1e-8))
        compressible_ns_num_variables: int = int(
            cns_data.get("num_variables", 3)
        )

        # --- 2D fluid data ---
        fluid_data: Dict[str, Any] = data_section.get("fluid_2d", {})
        fluid_2d_nt: int = int(fluid_data.get("nt", 32))
        fluid_2d_nx: int = int(fluid_data.get("nx", 64))
        fluid_2d_ny: int = int(fluid_data.get("ny", 64))
        fluid_2d_num_control_vars: int = int(
            fluid_data.get("num_control_vars_per_step", 3584)
        )

        # --- Ablation study ---
        ablation_section: Dict[str, Any] = d.get("ablation", {})
        ablation_noise_scale_factors: List[float] = [
            float(v)
            for v in ablation_section.get(
                "noise_scale_factors", [0.0001, 0.001, 0.01]
            )
        ]
        ablation_data_size_fractions: List[float] = [
            float(v)
            for v in ablation_section.get(
                "data_size_fractions", [0.2, 0.4, 0.6, 0.8]
            )
        ]
        ablation_train_size: int = int(
            ablation_section.get("ablation_train_size", 9000)
        )
        ablation_ddim_step_values: List[int] = [
            int(v)
            for v in ablation_section.get(
                "ddim_step_values", [20, 40, 50, 100, 200]
            )
        ]
        ablation_ddim_eta_values: List[float] = [
            float(v)
            for v in ablation_section.get(
                "ddim_eta_values", [0.2, 0.5, 0.8, 1.0]
            )
        ]
        ablation_wavelet_types: List[str] = [
            str(v)
            for v in ablation_section.get(
                "wavelet_types_to_compare", ["bior1.3", "bior2.4", "db4", "sym4"]
            )
        ]
        ablation_control_noise_prob: float = float(
            ablation_section.get("control_noise_prob", 0.1)
        )

        return cls(
            experiment=experiment,
            mode=mode,
            spatial_dim=spatial_dim,
            seed=seed,
            device=device,
            wavelet_type=wavelet_type,
            wavelet_mode=wavelet_mode,
            wavelet_level=wavelet_level,
            wavelet_transform_dim=wavelet_transform_dim,
            wavelet_library=wavelet_library,
            num_diffusion_timesteps=num_diffusion_timesteps,
            beta_schedule=beta_schedule,
            cfg_dropout_prob=cfg_dropout_prob,
            cfg_weight=cfg_weight,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
            guidance_lambda=guidance_lambda,
            guidance_schedule=guidance_schedule,
            unet_init_dim=unet_init_dim,
            unet_dim_mults=unet_dim_mults,
            unet_resnet_groups=unet_resnet_groups,
            unet_attn_heads=unet_attn_heads,
            unet_attn_head_dim=unet_attn_head_dim,
            unet_n_downsample=unet_n_downsample,
            use_dual_unet=use_dual_unet,
            conv3d_kernel=conv3d_kernel,
            conv3d_padding=conv3d_padding,
            conv3d_stride=conv3d_stride,
            downsample_kernel=downsample_kernel,
            downsample_padding=downsample_padding,
            downsample_stride=downsample_stride,
            upsample_kernel=upsample_kernel,
            upsample_padding=upsample_padding,
            upsample_stride=upsample_stride,
            batch_size=batch_size,
            lr=lr,
            train_steps=train_steps,
            lr_scheduler=lr_scheduler,
            num_gpus=num_gpus,
            num_sr_levels=num_sr_levels,
            eval_interp_modes=eval_interp_modes,
            data_path=data_path,
            checkpoint_dir=checkpoint_dir,
            results_dir=results_dir,
            nu=nu,
            T=T,
            nx_fine=nx_fine,
            nt_fine=nt_fine,
            nx_coarse=nx_coarse,
            nt_coarse=nt_coarse,
            num_train=num_train,
            num_test_control=num_test_control,
            num_test_sr_base=num_test_sr_base,
            num_test_sr_levels=num_test_sr_levels,
            control_alpha=control_alpha,
            solver_nx=solver_nx,
            solver_nt=solver_nt,
            compressible_ns_hdf5_filename=compressible_ns_hdf5_filename,
            compressible_ns_eta=compressible_ns_eta,
            compressible_ns_zeta=compressible_ns_zeta,
            compressible_ns_num_variables=compressible_ns_num_variables,
            fluid_2d_nt=fluid_2d_nt,
            fluid_2d_nx=fluid_2d_nx,
            fluid_2d_ny=fluid_2d_ny,
            fluid_2d_num_control_vars=fluid_2d_num_control_vars,
            ablation_noise_scale_factors=ablation_noise_scale_factors,
            ablation_data_size_fractions=ablation_data_size