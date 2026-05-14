## config.py
"""Configuration module for LUNO: Linearization Turns Neural Operators into
Function-Valued Gaussian Processes.

This module defines the Config dataclass that serves as the single source of
truth for all hyperparameters. All other modules import Config from here.
Values are taken directly from the paper where specified; assumed defaults are
clearly marked.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class Config:
    """Flat configuration dataclass for the LUNO reproduction.

    All fields have explicit types and default values. Fields marked with
    [ASSUMED] in comments are not explicitly stated in the paper and use
    standard defaults.

    The from_dict() classmethod translates a nested YAML dict (as loaded by
    PyYAML) into this flat dataclass. The to_dict() method reconstructs the
    nested structure for serialization.
    """

    # -----------------------------------------------------------------------
    # FNO Architecture
    # Paper Section 5 / Appendix D.2
    # -----------------------------------------------------------------------
    fno_modes: int = 12
    """k_max: number of Fourier modes per spatial dimension. Paper: '12 modes'."""

    fno_channels: int = 18
    """d_v': hidden channel width, constant throughout. Paper: '18 hidden dimensions'."""

    fno_blocks: int = 4
    """Total number of Fourier blocks L. Paper: '4 Fourier blocks'.
    LUNO uses the last block (index L-1 = 3) for last-layer inference."""

    activation: str = "gelu"
    """[ASSUMED] Activation function for FNO blocks. Standard FNO uses GELU."""

    input_steps: int = 10
    """Number of input time steps fed to the FNO. Paper: '10 initial time steps'."""

    spatial_padding: int = 2
    """Zero-padding at spatial borders. Paper: 'padded by two constant zero grid points'."""

    out_channels: int = 1
    """Number of output channels (predicting one time step ahead)."""

    # -----------------------------------------------------------------------
    # Training
    # Paper Appendix D.2
    # -----------------------------------------------------------------------
    lr: float = 1.0e-3
    """[ASSUMED] Learning rate for AdamW. Standard FNO default."""

    weight_decay: float = 1.0e-4
    """[ASSUMED] Weight decay for AdamW. Standard default."""

    warmup_fraction: float = 0.05
    """[ASSUMED] Fraction of total epochs used for LR warmup. 5% assumed."""

    low_data_epochs: int = 100
    """Training epochs for low-data regime. Paper: 'trained for 100 epochs'."""

    full_epochs: int = 1000
    """Training epochs for OOD/full regime. Paper: 'trained for 1000 epochs'."""

    # -----------------------------------------------------------------------
    # Dataset: Low-Data Regime (1D PDEs via APEBench)
    # Paper Appendix D.1.1 / Table 3
    # -----------------------------------------------------------------------
    low_data_train_traj: int = 25
    """Training trajectories for low-data regime. Paper Table 3."""

    spatial_res_1d: int = 256
    """Spatial resolution for 1D PDEs. Paper Table 3."""

    time_steps: int = 59
    """Temporal resolution (time steps per trajectory). Paper Table 3."""

    # -----------------------------------------------------------------------
    # Dataset: OOD Regime (2D Advection-Diffusion-Reaction)
    # Paper Appendix D.1.2
    # -----------------------------------------------------------------------
    ood_train_traj: int = 1000
    """Training trajectories for OOD regime. Paper Appendix D.1.2."""

    spatial_res_2d: int = 100
    """Spatial resolution for 2D PDEs (100x100 grid). Paper Appendix D.1.2."""

    ood_alpha: float = 0.026
    """Diffusion coefficient, held constant. Paper Appendix D.1.2."""

    ood_dt: float = 5.0e-10
    """Time step for OOD PDE solver. Paper Appendix D.1.2."""

    ood_n_steps_raw: int = 200
    """Total integration steps before subsampling. Paper Appendix D.1.2."""

    ood_variants: List[str] = dataclasses.field(
        default_factory=lambda: ["base", "flip", "pos", "pos_neg", "pos_neg_flip"]
    )
    """OOD dataset variants. Paper Appendix D.1.2."""

    # Initial condition parameters for OOD data generation
    n_blobs_min: int = 1
    """Minimum number of Gaussian blobs in initial condition. Paper: '1-10 blobs'."""

    n_blobs_max: int = 10
    """Maximum number of Gaussian blobs in initial condition. Paper: '1-10 blobs'."""

    blob_scale_min: float = 5.0
    """[ASSUMED] Minimum blob scale in grid points. Approximated from Figure 5."""

    blob_scale_max: float = 15.0
    """[ASSUMED] Maximum blob scale in grid points. Approximated from Figure 5."""

    blob_amplitude_min: float = 0.5
    """[ASSUMED] Minimum blob amplitude. Approximated from Figure 5."""

    blob_amplitude_max: float = 2.0
    """[ASSUMED] Maximum blob amplitude. Approximated from Figure 5."""

    velocity_range_min: float = -1.0
    """[ASSUMED] Minimum velocity field component value."""

    velocity_range_max: float = 1.0
    """[ASSUMED] Maximum velocity field component value."""

    # -----------------------------------------------------------------------
    # Validation and Test Set Sizes
    # Paper Appendix D.5 / Section 5
    # -----------------------------------------------------------------------
    n_val_pairs: int = 250
    """Validation set size for calibration. Paper Appendix D.5."""

    n_test_pairs: int = 250
    """Test set size for evaluation. Paper Section 5."""

    # -----------------------------------------------------------------------
    # Uncertainty Quantification
    # Paper Appendix D.3
    # -----------------------------------------------------------------------
    ggn_rank: int = 500
    """Low-rank GGN approximation rank. Paper Appendix D.3.4: 'low rank of 500'."""

    ggn_last_layer_only: bool = True
    """Restrict GGN to last Fourier block only. Paper Section 3.2.1 / Appendix C.1."""

    ggn_n_pairs_low_data: int = 25
    """GGN data pairs for low-data regime. Paper Appendix D.3.4: 'all input-output pairs'."""

    ggn_n_pairs_ood: int = 1000
    """GGN data pairs for OOD regime. Paper Appendix D.3.4: 'minibatch of 1000 pairs'."""

    n_samples: int = 200
    """Number of weight samples for Sample-* methods. Paper Appendix D.3.5."""

    n_ensemble: int = 10
    """Number of ensemble members. Paper Section 5 / Appendix D.3.2."""

    # -----------------------------------------------------------------------
    # Calibration
    # Paper Appendix D.5
    # -----------------------------------------------------------------------
    cal_grid_size: int = 500
    """Calibration grid size. Paper Appendix D.5: 'logarithmically spaced grid with 500 points'."""

    cal_grid_range_factor: float = 100.0
    """Grid spans [center/factor, center*factor] in log space."""

    cal_prior_prec_center: float = 1.0
    """[ASSUMED] Starting center for LA prior precision grid search."""

    cal_sigma_sq_iso_center: float = 1.0
    """[ASSUMED] Starting center for isotropic variance grid search."""

    cal_sigma_perturb_center: float = 0.01
    """[ASSUMED] Starting center for input perturbation sigma grid search."""

    # -----------------------------------------------------------------------
    # Experiment Settings
    # -----------------------------------------------------------------------
    experiment: str = "low_data"
    """Experiment mode. Options: 'low_data' or 'ood'."""

    pde_name: str = "burgers"
    """PDE name for low_data mode. Options: 'burgers', 'hyper_diffusion', 'ks_conservative'."""

    seed: int = 42
    """Global random seed."""

    output_dir: str = "outputs"
    """Base output directory for results."""

    # -----------------------------------------------------------------------
    # Logging and Checkpointing
    # -----------------------------------------------------------------------
    log_every_n_epochs: int = 10
    """[ASSUMED] Log training loss every N epochs."""

    save_checkpoints: bool = True
    """[ASSUMED] Whether to save model checkpoints."""

    checkpoint_dir: str = "checkpoints"
    """[ASSUMED] Directory for model checkpoints."""

    results_dir: str = "results"
    """[ASSUMED] Directory for evaluation results."""

    def __post_init__(self) -> None:
        """Validate configuration values after construction."""
        valid_experiments = {"low_data", "ood"}
        if self.experiment not in valid_experiments:
            raise ValueError(
                f"experiment must be one of {valid_experiments}, "
                f"got '{self.experiment}'"
            )

        valid_pde_names = {"burgers", "hyper_diffusion", "ks_conservative"}
        if self.pde_name not in valid_pde_names:
            raise ValueError(
                f"pde_name must be one of {valid_pde_names}, "
                f"got '{self.pde_name}'"
            )

        if self.fno_blocks < 2:
            raise ValueError(
                f"fno_blocks must be >= 2 (need at least one intermediate block "
                f"+ one last block for LUNO), got {self.fno_blocks}"
            )

        if self.ggn_rank <= 0:
            raise ValueError(f"ggn_rank must be > 0, got {self.ggn_rank}")

        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be > 0, got {self.n_samples}")

        if self.n_ensemble <= 0:
            raise ValueError(f"n_ensemble must be > 0, got {self.n_ensemble}")

        if self.fno_modes <= 0:
            raise ValueError(f"fno_modes must be > 0, got {self.fno_modes}")

        if self.fno_channels <= 0:
            raise ValueError(f"fno_channels must be > 0, got {self.fno_channels}")

        if not (0.0 < self.warmup_fraction < 1.0):
            raise ValueError(
                f"warmup_fraction must be in (0, 1), got {self.warmup_fraction}"
            )

        if self.cal_grid_size <= 0:
            raise ValueError(
                f"cal_grid_size must be > 0, got {self.cal_grid_size}"
            )

        if self.cal_grid_range_factor <= 1.0:
            raise ValueError(
                f"cal_grid_range_factor must be > 1.0, got {self.cal_grid_range_factor}"
            )

        valid_activations = {"gelu", "relu", "tanh", "silu"}
        if self.activation not in valid_activations:
            raise ValueError(
                f"activation must be one of {valid_activations}, "
                f"got '{self.activation}'"
            )

    # -----------------------------------------------------------------------
    # Computed Properties
    # -----------------------------------------------------------------------

    @property
    def warmup_epochs(self) -> int:
        """Compute warmup epochs as a fraction of total training epochs.

        Returns:
            Number of warmup epochs (at least 1).
        """
        total: int = self.epochs
        return max(1, int(self.warmup_fraction * total))

    @property
    def epochs(self) -> int:
        """Return the appropriate epoch count for the current experiment mode.

        Returns:
            low_data_epochs for 'low_data', full_epochs for 'ood'.
        """
        if self.experiment == "low_data":
            return self.low_data_epochs
        return self.full_epochs

    @property
    def n_train_traj(self) -> int:
        """Return the appropriate training trajectory count for the current mode.

        Returns:
            low_data_train_traj for 'low_data', ood_train_traj for 'ood'.
        """
        if self.experiment == "low_data":
            return self.low_data_train_traj
        return self.ood_train_traj

    @property
    def in_channels(self) -> int:
        """Return the appropriate input channel count for the current experiment.

        For 1D PDEs: input_steps + 1 (velocity placeholder) + 1 (reaction placeholder).
        For 2D PDEs: input_steps + 2 (velocity x, y) + 1 (reaction).

        Returns:
            Number of input channels for the FNO.
        """
        if self.experiment == "low_data":
            return self.input_steps + 2  # 10 + velocity_placeholder + reaction_placeholder
        return self.input_steps + 3  # 10 + vx + vy + reaction

    @property
    def spatial_res(self) -> int:
        """Return the appropriate spatial resolution for the current experiment.

        Returns:
            spatial_res_1d for 'low_data', spatial_res_2d for 'ood'.
        """
        if self.experiment == "low_data":
            return self.spatial_res_1d
        return self.spatial_res_2d

    @property
    def ggn_n_pairs(self) -> int:
        """Return the appropriate GGN data pair count for the current experiment.

        Returns:
            ggn_n_pairs_low_data for 'low_data', ggn_n_pairs_ood for 'ood'.
        """
        if self.experiment == "low_data":
            return self.ggn_n_pairs_low_data
        return self.ggn_n_pairs_ood

    @property
    def apebench_scenario_name(self) -> str:
        """Return the APEBench scenario name for the current PDE.

        Returns:
            APEBench scenario identifier string.

        Raises:
            KeyError: If pde_name is not in the mapping (prevented by __post_init__).
        """
        mapping: Dict[str, str] = {
            "burgers": "burgers_1d",
            "hyper_diffusion": "hyper_diffusion_1d",
            "ks_conservative": "ks_conservative_1d",
        }
        return mapping[self.pde_name]

    @property
    def is_1d(self) -> bool:
        """Return True if the current experiment uses 1D PDEs.

        Returns:
            True for 'low_data', False for 'ood'.
        """
        return self.experiment == "low_data"

    @property
    def is_2d(self) -> bool:
        """Return True if the current experiment uses 2D PDEs.

        Returns:
            True for 'ood', False for 'low_data'.
        """
        return self.experiment == "ood"

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """Construct a Config from a nested dict (e.g., loaded from YAML).

        Maps nested YAML keys to flat dataclass fields. Missing keys fall back
        to the dataclass defaults.

        Args:
            d: Nested dictionary, typically loaded from config.yaml.

        Returns:
            A fully constructed and validated Config instance.
        """
        exp_cfg: Dict[str, Any] = d.get("experiment", {})
        model_cfg: Dict[str, Any] = d.get("model", {})
        training_cfg: Dict[str, Any] = d.get("training", {})
        data_cfg: Dict[str, Any] = d.get("data", {})
        low_data_cfg: Dict[str, Any] = data_cfg.get("low_data", {})
        ood_cfg: Dict[str, Any] = data_cfg.get("ood", {})
        ic_cfg: Dict[str, Any] = ood_cfg.get("ic", {})
        vel_cfg: Dict[str, Any] = ood_cfg.get("velocity", {})
        uncertainty_cfg: Dict[str, Any] = d.get("uncertainty", {})
        ggn_cfg: Dict[str, Any] = uncertainty_cfg.get("ggn", {})
        sampling_cfg: Dict[str, Any] = uncertainty_cfg.get("sampling", {})
        ensemble_cfg: Dict[str, Any] = uncertainty_cfg.get("ensemble", {})
        cal_cfg: Dict[str, Any] = d.get("calibration", {})
        eval_cfg: Dict[str, Any] = d.get("evaluation", {})
        logging_cfg: Dict[str, Any] = d.get("logging", {})

        # Extract OOD variants, handling both list and nested dict formats
        ood_variants_raw = ood_cfg.get("variants", ["base", "flip", "pos", "pos_neg", "pos_neg_flip"])
        if isinstance(ood_variants_raw, list):
            ood_variants: List[str] = [str(v) for v in ood_variants_raw]
        else:
            ood_variants = ["base", "flip", "pos", "pos_neg", "pos_neg_flip"]

        return cls(
            # FNO Architecture
            fno_modes=int(model_cfg.get("modes", 12)),
            fno_channels=int(model_cfg.get("channels", 18)),
            fno_blocks=int(model_cfg.get("n_blocks", 4)),
            activation=str(model_cfg.get("activation", "gelu")),
            input_steps=int(model_cfg.get("input_steps", 10)),
            spatial_padding=int(model_cfg.get("spatial_padding", 2)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            # Training
            lr=float(training_cfg.get("learning_rate", 1.0e-3)),
            weight_decay=float(training_cfg.get("weight_decay", 1.0e-4)),
            warmup_fraction=float(training_cfg.get("warmup_fraction", 0.05)),
            low_data_epochs=int(training_cfg.get("low_data_epochs", 100)),
            full_epochs=int(training_cfg.get("full_epochs", 1000)),
            # Low-data dataset
            low_data_train_traj=int(low_data_cfg.get("n_train", 25)),
            spatial_res_1d=int(low_data_cfg.get("spatial_res", 256)),
            time_steps=int(low_data_cfg.get("time_steps", 59)),
            # OOD dataset
            ood_train_traj=int(ood_cfg.get("n_train", 1000)),
            spatial_res_2d=int(ood_cfg.get("spatial_res", 100)),
            ood_alpha=float(ood_cfg.get("alpha", 0.026)),
            ood_dt=float(ood_cfg.get("dt", 5.0e-10)),
            ood_n_steps_raw=int(ood_cfg.get("n_steps_raw", 200)),
            ood_variants=ood_variants,
            # IC parameters
            n_blobs_min=int(ic_cfg.get("n_blobs_min", 1)),
            n_blobs_max=int(ic_cfg.get("n_blobs_max", 10)),
            blob_scale_min=float(ic_cfg.get("blob_scale_min", 5.0)),
            blob_scale_max=float(ic_cfg.get("blob_scale_max", 15.0)),
            blob_amplitude_min=float(ic_cfg.get("blob_amplitude_min", 0.5)),
            blob_amplitude_max=float(ic_cfg.get("blob_amplitude_max", 2.0)),
            velocity_range_min=float(vel_cfg.get("range_min", -1.0)),
            velocity_range_max=float(vel_cfg.get("range_max", 1.0)),
            # Validation and test sizes
            n_val_pairs=int(cal_cfg.get("n_val_pairs", 250)),
            n_test_pairs=int(eval_cfg.get("n_test_pairs", 250)),
            # GGN
            ggn_rank=int(ggn_cfg.get("rank", 500)),
            ggn_last_layer_only=bool(ggn_cfg.get("last_layer_only", True)),
            ggn_n_pairs_low_data=int(ggn_cfg.get("n_pairs_low_data", 25)),
            ggn_n_pairs_ood=int(ggn_cfg.get("n_pairs_ood", 1000)),
            # Sampling and ensemble
            n_samples=int(sampling_cfg.get("n_samples", 200)),
            n_ensemble=int(ensemble_cfg.get("n_members", 10)),
            # Calibration
            cal_grid_size=int(cal_cfg.get("grid_size", 500)),
            cal_grid_range_factor=float(cal_cfg.get("grid_range_factor", 100.0)),
            cal_prior_prec_center=float(cal_cfg.get("prior_prec_center", 1.0)),
            cal_sigma_sq_iso_center=float(cal_cfg.get("sigma_sq_iso_center", 1.0)),
            cal_sigma_perturb_center=float(cal_cfg.get("sigma_perturb_center", 0.01)),
            # Experiment
            experiment=str(exp_cfg.get("mode", "low_data")),
            pde_name=str(exp_cfg.get("pde_name", "burgers")),
            seed=int(exp_cfg.get("seed", 42)),
            output_dir=str(exp_cfg.get("output_dir", "outputs")),
            # Logging
            log_every_n_epochs=int(logging_cfg.get("log_every_n_epochs", 10)),
            save_checkpoints=bool(logging_cfg.get("save_checkpoints", True)),
            checkpoint_dir=str(logging_cfg.get("checkpoint_dir", "checkpoints")),
            results_dir=str(logging_cfg.get("results_dir", "results")),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the Config back to a nested dict matching the YAML structure.

        Returns:
            Nested dictionary suitable for YAML serialization.
        """
        return {
            "experiment": {
                "mode": self.experiment,
                "pde_name": self.pde_name,
                "seed": self.seed,
                "output_dir": self.output_dir,
            },
            "model": {
                "modes": self.fno_modes,
                "channels": self.fno_channels,
                "n_blocks": self.fno_blocks,
                "activation": self.activation,
                "input_steps": self.input_steps,
                "spatial_padding": self.spatial_padding,
                "out_channels": self.out_channels,
            },
            "training": {
                "low_data_epochs": self.low_data_epochs,
                "full_epochs": self.full_epochs,
                "loss": "mse",
                "optimizer": "adamw",
                "learning_rate": self.lr,
                "weight_decay": self.weight_decay,
                "lr_schedule": "cosine_decay_with_warmup",
                "warmup_fraction": self.warmup_fraction,
                "batch_size_per_epoch": 1,
            },
            "data": {
                "low_data": {
                    "spatial_res": self.spatial_res_1d,
                    "time_steps": self.time_steps,
                    "n_train": self.low_data_train_traj,
                    "n_val": self.n_val_pairs,
                    "n_test": self.n_test_pairs,
                    "pde_configs": {
                        "burgers": {"apebench_name": "burgers_1d"},
                        "hyper_diffusion": {"apebench_name": "hyper_diffusion_1d"},
                        "ks_conservative": {"apebench_name": "ks_conservative_1d"},
                    },
                },
                "ood": {
                    "spatial_res": self.spatial_res_2d,
                    "time_steps": self.time_steps,
                    "n_steps_raw": self.ood_n_steps_raw,
                    "dt": self.ood_dt,
                    "alpha": self.ood_alpha,
                    "n_train": self.ood_train_traj,
                    "n_val": self.n_val_pairs,
                    "n_test": self.n_test_pairs,
                    "variants": self.ood_variants,
                    "ic": {
                        "n_blobs_min": self.n_blobs_min,
                        "n_blobs_max": self.n_blobs_max,
                        "blob_scale_min": self.blob_scale_min,
                        "blob_scale_max": self.blob_scale_max,
                        "blob_amplitude_min": self.blob_amplitude_min,
                        "blob_amplitude_max": self.blob_amplitude_max,
                    },
                    "velocity": {
                        "range_min": self.velocity_range_min,
                        "range_max": self.velocity_range_max,
                    },
                },
            },
            "uncertainty": {
                "ggn": {
                    "rank": self.ggn_rank,
                    "last_layer_only": self.ggn_last_layer_only,
                    "n_pairs_low_data": self.ggn_n_pairs_low_data,
                    "n_pairs_ood": self.ggn_n_pairs_ood,
                },
                "sampling": {
                    "n_samples": self.n_samples,
                },
                "ensemble": {
                    "n_members": self.n_ensemble,
                },
            },
            "calibration": {
                "n_val_pairs": self.n_val_pairs,
                "metric": "marginal_nll",
                "grid_size": self.cal_grid_size,
                "grid_range_factor": self.cal_grid_range_factor,
                "prior_prec_center": self.cal_prior_prec_center,
                "sigma_sq_iso_center": self.cal_sigma_sq_iso_center,
                "sigma_perturb_center": self.cal_sigma_perturb_center,
            },
            "evaluation": {
                "n_test_pairs": self.n_test_pairs,
                "metrics": ["rmse", "nll", "chi2"],
                "rollout": {
                    "n_trajectories": 50,
                    "n_steps": 59,
                    "dataset": "pos_neg_flip",
                },
            },
            "logging": {
                "log_every_n_epochs": self.log_every_n_epochs,
                "save_checkpoints": self.save_checkpoints,
                "checkpoint_dir": self.checkpoint_dir,
                "results_dir": self.results_dir,
            },
        }

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load a Config from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A fully constructed and validated Config instance.

        Raises:
            ImportError: If PyYAML is not installed.
            FileNotFoundError: If the YAML file does not exist.
        """
        try:
            import yaml  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load config from YAML. "
                "Install it with: pip install pyyaml"
            ) from exc

        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)

        return cls.from_dict(raw)

    def save_yaml(self, path: str) -> None:
        """Save the Config to a YAML file.

        Args:
            path: Path where the YAML file will be written.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        try:
            import yaml  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to save config to YAML. "
                "Install it with: pip install pyyaml"
            ) from exc

        import os

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        """Return a concise string representation of the Config."""
        return (
            f"Config("
            f"experiment='{self.experiment}', "
            f"pde_name='{self.pde_name}', "
            f"fno_modes={self.fno_modes}, "
            f"fno_channels={self.fno_channels}, "
            f"fno_blocks={self.fno_blocks}, "
            f"epochs={self.epochs}, "
            f"n_train_traj={self.n_train_traj}, "
            f"ggn_rank={self.ggn_rank}"
            f")"
        )
