"""Configuration for Robotic World Model (RWM) and MBPO-PPO."""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class RobotSpec:
    """Specification for a robot's observation, action, and privileged info spaces."""

    obs_dim: int
    action_dim: int
    privileged_dim: int
    policy_obs_dim: int
    num_joints: int
    num_feet: int = 4

    # Index slices for observation components
    obs_base_lin_vel: Tuple[int, int] = (0, 3)
    obs_base_ang_vel: Tuple[int, int] = (3, 6)
    obs_projected_gravity: Tuple[int, int] = (6, 9)
    obs_joint_pos_start: int = 9
    obs_joint_vel_start: Optional[int] = None
    obs_joint_tau_start: Optional[int] = None


ANYMAL_D_SPEC = RobotSpec(
    obs_dim=45,
    action_dim=12,
    privileged_dim=8,
    policy_obs_dim=48,
    num_joints=12,
    num_feet=4,
    obs_base_lin_vel=(0, 3),
    obs_base_ang_vel=(3, 6),
    obs_projected_gravity=(6, 9),
    obs_joint_pos_start=9,
    obs_joint_vel_start=21,
    obs_joint_tau_start=33,
)

UNITREE_G1_SPEC = RobotSpec(
    obs_dim=96,
    action_dim=29,
    privileged_dim=30,
    policy_obs_dim=99,
    num_joints=29,
    num_feet=2,
    obs_base_lin_vel=(0, 3),
    obs_base_ang_vel=(3, 6),
    obs_projected_gravity=(6, 9),
    obs_joint_pos_start=9,
    obs_joint_vel_start=38,
    obs_joint_tau_start=67,
)


@dataclass
class RWMArchConfig:
    """Architecture configuration for RWM (Table S7)."""

    gru_hidden_size: int = 256
    gru_num_layers: int = 2  # hidden shape 256, 256 → 2 layers
    head_hidden_size: int = 128
    head_activation: str = "relu"


@dataclass
class PolicyArchConfig:
    """Architecture configuration for PPO policy and value function (Table S9)."""

    hidden_shape: Tuple[int, ...] = (128, 128, 128)
    activation: str = "elu"


@dataclass
class MLPBaselineConfig:
    """MLP baseline architecture (Table S8)."""

    hidden_shape: Tuple[int, ...] = (256, 256)
    activation: str = "relu"


@dataclass
class RSSMBaselineConfig:
    """RSSM baseline architecture (Table S8)."""

    rnn_type: str = "gru"
    hidden_size: int = 256
    num_layers: int = 2
    latent_dim: int = 64
    prior_type: str = "categorical"
    num_categories: int = 32


@dataclass
class TransformerBaselineConfig:
    """Transformer baseline architecture (Table S8)."""

    model_type: str = "decoder"
    d_model: int = 64
    nhead: int = 8
    num_layers: int = 2
    context_length: int = 32
    positional_encoding: str = "sinusoidal"


@dataclass
class RWMConfig:
    """Full RWM configuration combining architecture and training parameters."""

    robot: RobotSpec = field(default_factory=ANYMAL_D_SPEC)
    arch: RWMArchConfig = field(default_factory=RWMArchConfig)

    # Training parameters (Table S10)
    dt: float = 0.02
    max_iterations: int = 2500
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 1024
    history_horizon: int = 32  # M
    forecast_horizon: int = 8  # N
    forecast_decay: float = 1.0  # α
    num_seeds: int = 5
    grad_clip: float = 10.0

    # Optimizer
    optimizer: str = "adam"


@dataclass
class MBPOPPOConfig:
    """Configuration for MBPO-PPO training (Table S11)."""

    robot: RobotSpec = field(default_factory=ANYMAL_D_SPEC)
    policy_arch: PolicyArchConfig = field(default_factory=PolicyArchConfig)

    imagination_envs: int = 4096
    imagination_steps_per_iteration: int = 100  # T
    dt: float = 0.02
    buffer_size: int = 1000  # |D|
    max_iterations: int = 2500
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    learning_epochs: int = 5
    mini_batches: int = 4
    kl_target: float = 0.01
    discount_factor: float = 0.99  # γ
    clip_range: float = 0.2  # ε
    entropy_coef: float = 0.005
    num_seeds: int = 5
    gae_lambda: float = 0.95
    value_loss_coef: float = 0.5


@dataclass
class RewardWeights:
    """Reward function weights (Table S6)."""

    w_vxy: float = 1.0
    w_omega_z: float = 0.5
    w_vz: float = -2.0
    w_omega_xy: float = -0.05
    w_q_tau: float = -2.5e-5
    w_q_ddot: float = -2.5e-7
    w_a_dot: float = -0.01
    w_f_air: float = 0.5
    w_undesired_contact: float = -1.0
    w_flat_orientation: float = -5.0
    w_foot_clearance: float = 0.0
    w_joint_deviation: float = 0.0

    # Temperature factors
    sigma_vxy: float = 0.25
    sigma_omega_z: float = 0.25


ANYMAL_D_REWARD = RewardWeights(
    w_vxy=1.0,
    w_omega_z=0.5,
    w_vz=-2.0,
    w_omega_xy=-0.05,
    w_q_tau=-2.5e-5,
    w_q_ddot=-2.5e-7,
    w_a_dot=-0.01,
    w_f_air=0.5,
    w_undesired_contact=-1.0,
    w_flat_orientation=-5.0,
    w_foot_clearance=0.0,
    w_joint_deviation=0.0,
)

UNITREE_G1_REWARD = RewardWeights(
    w_vxy=1.0,
    w_omega_z=0.5,
    w_vz=-2.0,
    w_omega_xy=-0.05,
    w_q_tau=-2.5e-5,
    w_q_ddot=-2.5e-7,
    w_a_dot=-0.05,
    w_f_air=0.0,
    w_undesired_contact=-1.0,
    w_flat_orientation=-5.0,
    w_foot_clearance=1.0,
    w_joint_deviation=-1.0,
)


@dataclass
class ExperimentConfig:
    """Master configuration for an experiment run."""

    rwm: RWMConfig = field(default_factory=RWMConfig)
    mbpo_ppo: MBPOPOConfig = field(default_factory=MBPOPOConfig)
    reward_weights: RewardWeights = field(default_factory=ANYMAL_D_REWARD)
    device: str = "cuda"
    seed: int = 0
    use_privileged: bool = True
    pretrain_rwm: bool = True
    pretrain_steps: int = 6_000_000  # 6M state transitions
    log_interval: int = 10
