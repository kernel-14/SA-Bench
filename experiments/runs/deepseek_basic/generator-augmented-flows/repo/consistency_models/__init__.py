"""
consistency_models - Implementation of "Improving Consistency Models with Generator-Augmented Flows"

Core components:
- model: Consistency model with skip-connection parameterization and SongUNet architecture
- coupling: IC, OT, and GC data-noise coupling strategies
- losses: CD, CT, and GC loss functions
- training: Training loops with EMA, Lion optimizer, joint learning
- scheduling: Noise schedules, discretization schedules, timestep distributions
- metrics: FID, KID, IS evaluation
- theory: Theoretical analysis (regularizer, transport cost)
"""

from .model import ConsistencyModel, model_parameterization, SongUNet
from .coupling import (
    independent_coupling, batch_ot_coupling, generator_augmented_coupling,
    compute_r_proxy,
)
from .losses import (
    consistency_distillation_loss,
    consistency_training_loss,
    gc_consistency_loss,
    joint_gc_loss,
    pseudo_huber_loss,
    get_distance_fn,
)
from .training import (
    train_consistency_model,
    train_gc_joint,
    ConsistencyTrainingConfig,
    EMA,
    LionOptimizer,
)
from .scheduling import (
    noise_schedule_karras,
    weighting_function,
    discretization_schedule,
    timestep_sampling_distribution,
    sample_timesteps,
    get_sigmas_for_indices,
)
from .metrics import (
    compute_fid,
    compute_kid,
    compute_is,
    InceptionV3FeatureExtractor,
    evaluate_model,
)
from .theory import (
    compute_regularizer_discrepancy,
    compute_transport_cost,
    compute_transport_cost_curve,
    compute_ic_transport_cost,
)

__version__ = "1.0.0"

__all__ = [
    # Model
    "ConsistencyModel",
    "model_parameterization",
    "SongUNet",
    # Coupling
    "independent_coupling",
    "batch_ot_coupling",
    "generator_augmented_coupling",
    "compute_r_proxy",
    # Losses
    "consistency_distillation_loss",
    "consistency_training_loss",
    "gc_consistency_loss",
    "joint_gc_loss",
    "pseudo_huber_loss",
    "get_distance_fn",
    # Training
    "train_consistency_model",
    "train_gc_joint",
    "ConsistencyTrainingConfig",
    "EMA",
    "LionOptimizer",
    # Scheduling
    "noise_schedule_karras",
    "weighting_function",
    "discretization_schedule",
    "timestep_sampling_distribution",
    "sample_timesteps",
    "get_sigmas_for_indices",
    # Metrics
    "compute_fid",
    "compute_kid",
    "compute_is",
    "InceptionV3FeatureExtractor",
    "evaluate_model",
    # Theory
    "compute_regularizer_discrepancy",
    "compute_transport_cost",
    "compute_transport_cost_curve",
    "compute_ic_transport_cost",
]
