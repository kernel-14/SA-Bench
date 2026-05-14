"""Configuration dataclasses and default configs for LUNO reproducibility."""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class FNOConfig:
    """Fourier Neural Operator architecture config (paper: 4 Fourier blocks, 12 modes, 18 hidden dims)."""
    n_modes: Tuple[int, ...] = (12,)
    hidden_dim: int = 18
    n_blocks: int = 4
    input_dim: int = 1
    output_dim: int = 1
    lifting_dim: Optional[int] = None

    def __post_init__(self):
        if self.lifting_dim is None:
            self.lifting_dim = self.hidden_dim


@dataclass
class TrainingConfig:
    """Training hyperparameters from paper."""
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    warmup_steps: int = 100
    batch_size: int = 1
    loss: str = "mse"
    optimizer: str = "adamw"

    # Cosine decay scheduler
    lr_schedule: str = "cosine_decay"


@dataclass
class LaplaceConfig:
    """Linearized Laplace approximation config."""
    prior_precision: float = 1.0
    low_rank: int = 500
    hessian_structure: str = "lowrank"  # lowrank or diag
    n_data_for_ggn: int = -1  # -1 means all data
    last_layer_only: bool = True


@dataclass
class DataConfig:
    """Dataset configuration matching APEBench paper settings."""
    # PDE type
    pde_name: str = "burgers"
    spatial_dim: int = 1
    # Spatial / temporal resolution
    spatial_resolution: int = 256
    temporal_resolution: int = 59
    # Train/val/test splits for low-data regime
    n_train_trajectories: int = 25
    n_val_trajectories: int = 250
    n_test_trajectories: int = 250
    # Time steps: 10 input, predict next 1
    n_input_steps: int = 10
    n_output_steps: int = 1
    # Domain
    domain_size: float = 1.0
    # OOD advection-diffusion params
    diffusion_coefficient: float = 0.026
    ode_dt: float = 5e-10
    n_time_steps_ode: int = 200


@dataclass
class UQConfig:
    """Uncertainty quantification method configuration."""
    method: str = "luno_la"  # one of: luno_iso, luno_la, sample_iso, sample_la, ensemble, input_perturbations
    # For sampling-based methods
    n_samples: int = 200
    # For isotropic Gaussian
    sigma2: float = 1.0
    # For input perturbations
    input_noise_sigma: float = 1.0
    # For Laplace
    laplace: LaplaceConfig = field(default_factory=LaplaceConfig)
    # For ensemble
    n_ensemble_members: int = 10
    # Calibration
    calibrate: bool = True
    calibration_grid_size: int = 500
    calibration_metric: str = "nll"


@dataclass
class ExperimentConfig:
    """Full configuration for an experiment."""
    fno: FNOConfig = field(default_factory=FNOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    uq: UQConfig = field(default_factory=UQConfig)
    seed: int = 42
    output_dir: str = "./outputs"


# Pre-defined configurations for different experiments
def get_low_data_burgers_config() -> ExperimentConfig:
    return ExperimentConfig(
        fno=FNOConfig(n_modes=(12,), hidden_dim=18, n_blocks=4, input_dim=11, output_dim=1),
        training=TrainingConfig(epochs=100, learning_rate=1e-3, batch_size=1),
        data=DataConfig(
            pde_name="burgers", spatial_dim=1,
            spatial_resolution=256, temporal_resolution=59,
            n_train_trajectories=25, n_val_trajectories=250, n_test_trajectories=250,
        ),
    )


def get_low_data_hyper_diffusion_config() -> ExperimentConfig:
    return ExperimentConfig(
        fno=FNOConfig(n_modes=(12,), hidden_dim=18, n_blocks=4, input_dim=11, output_dim=1),
        training=TrainingConfig(epochs=100, learning_rate=1e-3, batch_size=1),
        data=DataConfig(
            pde_name="hyper_diffusion", spatial_dim=1,
            spatial_resolution=256, temporal_resolution=59,
            n_train_trajectories=25, n_val_trajectories=250, n_test_trajectories=250,
        ),
    )


def get_low_data_kuramoto_sivashinsky_config() -> ExperimentConfig:
    return ExperimentConfig(
        fno=FNOConfig(n_modes=(12,), hidden_dim=18, n_blocks=4, input_dim=11, output_dim=1),
        training=TrainingConfig(epochs=100, learning_rate=1e-3, batch_size=1),
        data=DataConfig(
            pde_name="kuramoto_sivashinsky", spatial_dim=1,
            spatial_resolution=256, temporal_resolution=59,
            n_train_trajectories=25, n_val_trajectories=250, n_test_trajectories=250,
        ),
    )


def get_ood_advection_config() -> ExperimentConfig:
    return ExperimentConfig(
        fno=FNOConfig(n_modes=(12, 12), hidden_dim=18, n_blocks=4, input_dim=13, output_dim=1),
        training=TrainingConfig(epochs=1000, learning_rate=1e-3, batch_size=1),
        data=DataConfig(
            pde_name="advection_diffusion", spatial_dim=2,
            spatial_resolution=100, temporal_resolution=59,
            n_train_trajectories=1000, n_val_trajectories=250, n_test_trajectories=250,
            diffusion_coefficient=0.026,
        ),
    )
