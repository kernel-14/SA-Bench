"""Configuration and hyperparameters for all experiments."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class BinomialConfig:
    """Synthetic binomial experiment config (Section 5.1)."""
    n_calibration: int = 10
    K: int = 4
    alpha: float = 0.4
    B: float = 1.0
    beta: float = 0.95
    M_trials: int = 10000
    mc_samples: int = 100000
    lambda_grid_size: int = 200
    rcps_delta: float = 0.05


@dataclass
class HeteroskedasticConfig:
    """Synthetic heteroskedastic experiment config (Section 5.2)."""
    n_calibration: int = 200
    alpha: float = 0.1
    beta: float = 0.95
    M_trials: int = 10000
    mc_samples: int = 100000
    lambda_grid_size: int = 200
    rcps_delta: float = 0.05
    x_range: Tuple[float, float] = (0.0, 4.0)


@dataclass
class MSCOCOConfig:
    """MS-COCO multilabel classification experiment config (Section 5.3)."""
    n_calibration: int = 1000
    n_test: int = 3952
    alpha: float = 0.1
    beta: float = 0.95
    B: float = 1.0
    M_trials: int = 10000
    mc_samples: int = 100000
    lambda_grid_size: int = 200
    rcps_delta: float = 0.05
    coco_root: str = ""
    num_classes: int = 80


@dataclass
class ExperimentConfig:
    """Master configuration for all experiments."""
    binomial: BinomialConfig = field(default_factory=BinomialConfig)
    heteroskedastic: HeteroskedasticConfig = field(default_factory=HeteroskedasticConfig)
    ms_coco: MSCOCOConfig = field(default_factory=MSCOCOConfig)
    seed: int = 42
