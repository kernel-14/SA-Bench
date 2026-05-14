from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class FNOConfig:
    n_modes: int = 12
    d_v: int = 18
    n_layers: int = 4
    padding: int = 2
    activation: str = "gelu"
    projection_hidden: int = 128


@dataclass
class TrainConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    low_data_epochs: int = 100
    ood_epochs: int = 1000
    batch_size: int = 1
    seed: int = 0


@dataclass
class LowDataConfig:
    n_train: int = 25
    n_val: int = 250
    n_test: int = 250
    spatial_res: int = 256
    temporal_res: int = 59
    n_history: int = 10


@dataclass
class OODConfig:
    n_train: int = 1000
    n_val: int = 250
    n_test: int = 250
    spatial_res: int = 100
    temporal_res: int = 59
    n_history: int = 10
    diffusion_coeff: float = 0.026
    dt: float = 5e-10
    n_time_steps: int = 200
    n_subsample: int = 59


@dataclass
class LaplaceConfig:
    rank: int = 500
    n_data_ggn_low_data: int = 25
    n_data_ggn_ood: int = 1000


@dataclass
class CalibrationConfig:
    n_grid: int = 500
    sigma2_min: float = 1e-6
    sigma2_max: float = 1e2
    n_perturbation_samples: int = 200


@dataclass
class EvalConfig:
    n_test_pairs: int = 250
    n_ensemble: int = 10
    n_samples: int = 200


@dataclass
class ExperimentConfig:
    fno: FNOConfig = field(default_factory=FNOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    low_data: LowDataConfig = field(default_factory=LowDataConfig)
    ood: OODConfig = field(default_factory=OODConfig)
    laplace: LaplaceConfig = field(default_factory=LaplaceConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


DEFAULT_CONFIG = ExperimentConfig()
