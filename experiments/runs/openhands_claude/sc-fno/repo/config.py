"""
Configuration for all experiments in the SC-FNO paper.

Hyperparameters from Tables C.7 and C.8.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class FNOConfig:
    """FNO architecture hyperparameters (Table C.7)."""
    modes_t: int = 8
    modes_x: int = 8
    modes_y: int = 8
    width: int = 20
    n_layers: int = 4
    learning_rate: float = 1e-3
    n_epochs: int = 500


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Loss weights
    c1: float = 1.0   # weight for L_u (data loss)
    c2: float = 1.0   # weight for L_s (sensitivity loss)
    c3: float = 1.0   # weight for L_eq (PINN equation loss)
    alpha_pinn: float = 1.0  # weight for IC/BC terms in PINN loss

    # Sensitivity loss sampling
    n_spatial_samples: int = 10   # n < N spatial points sampled per epoch
    n_time_samples: int = 10      # t < T time points sampled per epoch

    # Data split
    train_frac: float = 0.70
    val_frac: float = 0.15
    seed: int = 42

    # Gradient computation
    use_ad: bool = True  # True: automatic differentiation, False: finite differences
    fd_eps: float = 1e-4  # finite difference step size


@dataclass
class ExperimentConfig:
    """Full experiment configuration for a specific equation."""
    name: str
    equation: str
    n_samples: int
    batch_size: int
    n_params: int
    fno: FNOConfig = field(default_factory=FNOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# ─── Per-equation configurations (Tables C.7, C.8) ───────────────────────────

ODE1_CONFIG = ExperimentConfig(
    name="ODE1",
    equation="ode1",
    n_samples=2000,
    batch_size=16,
    n_params=3,
    fno=FNOConfig(modes_t=8, modes_x=8, width=20, n_layers=4, learning_rate=1e-3, n_epochs=500),
)

ODE2_CONFIG = ExperimentConfig(
    name="ODE2",
    equation="ode2",
    n_samples=2000,
    batch_size=16,
    n_params=7,
    fno=FNOConfig(modes_t=8, modes_x=8, width=20, n_layers=4, learning_rate=1e-3, n_epochs=500),
)

PDE1_CONFIG = ExperimentConfig(
    name="PDE1",
    equation="pde1",
    n_samples=2000,
    batch_size=4,
    n_params=5,
    fno=FNOConfig(modes_t=8, modes_x=8, width=20, n_layers=4, learning_rate=1e-3, n_epochs=500),
)

PDE2_CONFIG = ExperimentConfig(
    name="PDE2",
    equation="pde2",
    n_samples=2000,
    batch_size=4,
    n_params=4,
    fno=FNOConfig(modes_t=8, modes_x=8, width=20, n_layers=4, learning_rate=1e-3, n_epochs=500),
)

PDE2_ZONED_100_CONFIG = ExperimentConfig(
    name="PDE2_Zoned_100",
    equation="pde2_zoned",
    n_samples=100,
    batch_size=1,
    n_params=82,  # 2*40 + 2
    fno=FNOConfig(modes_t=8, modes_x=8, width=20, n_layers=4, learning_rate=1e-3, n_epochs=500),
)

PDE2_ZONED_500_CONFIG = ExperimentConfig(
    name="PDE2_Zoned_500",
    equation="pde2_zoned",
    n_samples=500,
    batch_size=1,
    n_params=82,
    fno=FNOConfig(modes_t=8, modes_x=8, width=20, n_layers=4, learning_rate=1e-3, n_epochs=500),
)

PDE3_CONFIG = ExperimentConfig(
    name="PDE3",
    equation="pde3",
    n_samples=1000,
    batch_size=4,
    n_params=2,
    fno=FNOConfig(modes_t=8, modes_x=8, modes_y=8, width=20, n_layers=4, learning_rate=1e-3, n_epochs=500),
)

PDE4_100_CONFIG = ExperimentConfig(
    name="PDE4_100",
    equation="pde4",
    n_samples=100,
    batch_size=1,
    n_params=5,
    fno=FNOConfig(modes_t=8, modes_x=8, width=20, n_layers=4, learning_rate=1e-3, n_epochs=500),
)

PDE4_500_CONFIG = ExperimentConfig(
    name="PDE4_500",
    equation="pde4",
    n_samples=500,
    batch_size=1,
    n_params=5,
    fno=FNOConfig(modes_t=8, modes_x=8, width=20, n_layers=4, learning_rate=1e-3, n_epochs=500),
)

# ─── Model variant names ──────────────────────────────────────────────────────

MODEL_VARIANTS = ["FNO", "FNO-PINN", "SC-FNO", "SC-FNO-PINN"]

# ─── Parameter ranges (Table B.6) ────────────────────────────────────────────

PARAM_RANGES = {
    "ode1": {"alpha": (1.0, 3.0), "beta": (1.0, 3.0), "gamma": (0.0, 1.0)},
    "ode2": {
        "alpha": (0.02, 0.06),
        "beta": (0.01, 0.03),
        "gamma": (20.0, 60.0),
        "delta": (0.5, 1.5),
        "omega": (0.2, 0.6),
        "epsilon": (0.0, 0.2),
        "zeta": (0.0, 0.2),
    },
    "pde1": {
        "c": (0.0, 0.25),
        "alpha": (0.0, 0.1),
        "beta": (0.0, 0.25),
        "gamma": (0.0, 0.25),
        "omega": (0.0, 0.25),
    },
    "pde2": {
        "alpha": (0.1, 1.0),
        "gamma": (0.025, 0.25),
        "delta": (0.1, 0.5),
        "omega": (0.01, 0.1),
    },
    "pde3": {
        "alpha": (math.pi, 5 * math.pi),
        "beta": (math.pi, 5 * math.pi),
    },
    "pde4": {
        "c": (0.1, 0.9),
        "alpha": (0.01, 1.0),
        "beta": (0.01, 1.0),
        "omega": (5.0, 10.0),
        "epsilon": (0.01, 1.0),
    },
}

# ─── Perturbation ratios for robustness experiments ───────────────────────────

PERTURBATION_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# ─── Inversion experiment settings ───────────────────────────────────────────

INVERSION_CONFIG = {
    "n_iter": 1000,
    "lr": 1e-2,
    "optimizer": "adam",
}
