"""
Adjoint Matching: Fine-tuning Flow and Diffusion Models with Memoryless SOC

This package implements the core algorithms from:
"Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control"
by Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, Ricky T. Q. Chen (FAIR, Meta)

Main components:
- noise_schedules: Memoryless noise schedule and related utilities
- adjoint_matching: Adjoint Matching algorithm (Algorithm 1)
- baselines: DRaFT, ReFL, DPO, Continuous/Discrete Adjoint baselines
- models: Neural network architectures
- sde_simulation: SDE simulation utilities
- toy_experiment: 1D toy experiment (Figure 2)
"""

from .noise_schedules import (
    FlowMatchingSchedule,
    get_sigma_memoryless_fm,
    get_eta_fm,
    get_kappa_fm,
)

from .adjoint_matching import (
    AdjointMatchingTrainer,
    compute_lean_adjoint,
    adjoint_matching_loss_fm,
    select_gradient_timesteps,
)

from .baselines import (
    draft_loss,
    refl_loss,
    dpo_loss_fm,
    continuous_adjoint_loss,
    discrete_adjoint_loss,
)

from .models import (
    MLPVelocityModel,
    ConditionalMLPVelocityModel,
    LatentVelocityModel,
    SinusoidalTimeEmbedding,
)

__all__ = [
    # Noise schedules
    "FlowMatchingSchedule",
    "get_sigma_memoryless_fm",
    "get_eta_fm",
    "get_kappa_fm",
    # Adjoint Matching
    "AdjointMatchingTrainer",
    "compute_lean_adjoint",
    "adjoint_matching_loss_fm",
    "select_gradient_timesteps",
    # Baselines
    "draft_loss",
    "refl_loss",
    "dpo_loss_fm",
    "continuous_adjoint_loss",
    "discrete_adjoint_loss",
    # Models
    "MLPVelocityModel",
    "ConditionalMLPVelocityModel",
    "LatentVelocityModel",
    "SinusoidalTimeEmbedding",
]
